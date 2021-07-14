import os
import wandb
import sys
sys.path.append(os.environ["GRAPH_SUM"])
from src.ot_coarsening.dataset_management.cnndm.cnn_daily_mail import CNNDailyMail
from src.ot_coarsening.dataset_management.graph_constructor import CNNDailyMailGraphConstructor
from src.ot_coarsening.ot_coarsening import MultiLayerCoarsening, Coarsening
from src.ot_coarsening.u_net import GraphUNetCoarsening
from src.ot_coarsening.ot_coarsening_dense import Coarsening as DenseCoarsening
import src.ot_coarsening.evaluation as evaluation
import torch
import torch.nn.functional as F
from torch.nn import BCELoss
import time
import numpy as np
import argparse
from tqdm import tqdm
from torch_geometric.data import DataListLoader, DataLoader, DenseDataLoader as DenseLoader, Batch
from torch_geometric.nn import DataParallel, GCNConv
from torch.optim import Adam
import warnings
warnings.filterwarnings("ignore")

# setup arguments
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=250)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--lr_decay_factor', type=float, default=0.9)
parser.add_argument('--lr_decay_step_size', type=int, default=50)
parser.add_argument('--opt_iters', type=int, default=10)
parser.add_argument('--eps', type=float, default=1.0)
parser.add_argument('--ratio', type=float, default=0.1)
parser.add_argument('--train', action='store_true')
parser.add_argument('--num_output_sentences', type=int, default=3)
parser.add_argument('--dense', type=bool, default=False)
parser.add_argument('--no_extra_mlp', action='store_true')

args = parser.parse_args()

def num_graphs(data):
    if isinstance(data, list):
        return len(data)
    elif data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)

def validation_test(model, dataset):
    print("Running validation")
    model.eval()
    # make data loader
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, exclude_keys=["label", "tfidf"], drop_last=True)#, num_workers=4)
    # go through the validation set
    total_loss = 0
    for example_graph in tqdm(loader):
        example_graph.to(device)
        _, _, _, _, opt_loss, _ = model(example_graph, num_output_sentences=args.num_output_sentences)
        if opt_loss == 0.0:
            continue
        total_loss += opt_loss.item() * num_graphs(example_graph)

    # average
    return total_loss / len(loader.dataset)

#@profile_every(1)
def train_iteration(model, optimizer, loader):
    model.train()
    total_loss = 0
    for graph_index, example_graph in enumerate(tqdm(loader)):
        example_graph.to(device)
        optimizer.zero_grad()
        _, _, _, _, opt_loss, _ = model(example_graph, num_output_sentences=args.num_output_sentences)
        if opt_loss == 0.0:
            continue
        opt_loss.backward()
        total_loss += opt_loss.item() * num_graphs(example_graph)
        optimizer.step()

    # empty reserved memory
    return total_loss / len(loader.dataset)

def train(model, train_dataset, validation_dataset, save_dir="$GRAPH_SUM/src/ot_coarsening/model_cache", run_name=None):
    if run_name is None:
        raise Exception("run_name is None in train")
    dirpath = os.path.join(os.path.expandvars(save_dir), run_name)
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
    model_path = os.path.join(dirpath, "savedmodels_eps"+str(args.eps)+"_iter"+str(args.opt_iters)+".pt")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, exclude_keys=["label", "tfidf"], drop_last=True)#, num_workers=2)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=0.0001) # adding a negative weight regularizaiton such that it cannot be zero.

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    for epoch in tqdm(range(1, args.epochs + 1)):
        # Do a training iteration
        train_loss = train_iteration(model, optimizer, train_loader)
        # Get the unsupervised validation loss
        unsupervised_validation_loss = validation_test(model, validation_dataset)
        # Get the Rouge train loss
        rouge_validation_loss = rouge_validation(model, validation_dataset)
        # Get the Rouge validation loss 
        # Log the loss
        loss_dictionary = {
            "Unsupervised Train Loss": train_loss,
            "Unsupervised Validation Loss": unsupervised_validation_loss,
            "Rouge Validation Loss": rouge_validation_loss
        } 
        wandb.log(loss_dictionary)
        # save the model
        torch.save(model, model_path)
        

def rouge_validation(model, dataset):
    rouge_evaluations = evaluation.perform_rouge_evaluations(model, dataset, dense=args.dense)

def setup_logging():
    """
        Sets up logging for a model training and evaluation pass 
        with weights and biases.
    """
    # 1. Start a new run
    wandb.init(project='graph-summarization', entity='helblazer811')
    # 2. Save model inputs and hyperparameters
    config = wandb.config
    # 3. Log the config
    # 4. Return the run name
    return wandb.run.name 

class BasicSupervisedModel(torch.nn.Module):
    
    def __init__(self):
        super(BasicSupervisedModel, self).__init__()
        # implement two GCN layers
        self.input_dimensionality = 768
        self.hidden = 1024
        self.layer_one = GCNConv(self.input_dimensionality, self.hidden)
        self.layer_two = GCNConv(self.hidden, 1)
        self.supervised_loss = BCELoss()

    """ 
        Takes a graph as input and outputs a onehot vector predicting 
        important sentences. Assume no batch.
    """
    def forward(self, input_graph, num_output_sentences=3):
        # unpack input
        data_list = input_graph.to_data_list()
        loss = 0.0
        coarse_indices = []
        for data in data_list:
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
            edge_weight = edge_attr.squeeze().float()
            num_sentences = data.num_sentences
            y = data.y
            # run the layers
            x = self.layer_one(x, edge_index, edge_weight)
            output_attention = self.layer_two(x, edge_index, edge_weight).squeeze()
            assert output_attention.shape[0] == x.shape[0]
            # convert the attention to a onehot vector
            sentence_attention = output_attention[0:num_sentences]
            # get topk
            num_output_nodes = torch.nonzero(y >= 0).shape[0]
            if num_output_nodes == 0:
                num_output_nodes = 1
            topk_values, topk_indices = torch.topk(sentence_attention, num_output_nodes)
            cutoff = topk_values[-1]
            # get all coarse indices
            all_topk_values, all_topk_indices = torch.topk(output_attention, x.shape[0])
            coarse_inds = all_topk_indices[torch.nonzero(all_topk_values >= cutoff).squeeze()]
            coarse_indices.append(coarse_inds)
            # compute output 
            output_onehot = torch.sigmoid(sentence_attention)
            # convert label to a onehot vector
            label_indices = y[torch.nonzero(y > 0)].long().squeeze()
            if len(label_indices.shape) == 0:
                label_indices = label_indices[None]
            label_onehot = torch.zeros(num_sentences).to(device).float()
            label_onehot[label_indices] = 1.0
            # compute the loss
            loss += self.supervised_loss(output_onehot, label_onehot)
            
        return None, None, None, None, loss / len(data_list), coarse_indices
 
"""
    Main function for performing OTCoarsening on a DailyMail graph
"""
def main():
    # Setup logging
    run_name = setup_logging()
    # Setup CNNDailyMail data
    graph_constructor = CNNDailyMailGraphConstructor()
    train_dataset = CNNDailyMail(graph_constructor=graph_constructor, perform_processing=False, proportion_of_dataset=1.0, dense=args.dense)
    validation_dataset = CNNDailyMail(graph_constructor=graph_constructor, mode="val", perform_processing=False, proportion_of_dataset=1.0, dense=args.dense)
    # Initialize the model
    model = BasicSupervisedModel()
    model.to(device)
    # Run training
    train(model, train_dataset, validation_dataset, run_name=run_name)

if __name__ == "__main__":
    main()
