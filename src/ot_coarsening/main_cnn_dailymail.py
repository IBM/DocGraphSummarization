import os
import wandb
import sys
sys.path.append(os.environ["GRAPH_SUM"])
# from src.dataset_management.datasets import get_dataset
from src.ot_coarsening.dataset_management.cnn_daily_mail import CNNDailyMail
from src.ot_coarsening.dataset_management.graph_constructor import CNNDailyMailGraphConstructor
from src.ot_coarsening.ot_coarsening import MultiLayerCoarsening, Coarsening
import src.ot_coarsening.evaluation as evaluation
import torch
import torch.nn.functional as F
import time
import numpy as np
import argparse
from tqdm import tqdm
from torch_geometric.data import DataListLoader, DataLoader, DenseDataLoader as DenseLoader, Batch
from torch_geometric.nn import DataParallel
from torch.optim import Adam

from pytorch_memlab import profile, set_target_gpu, profile_every
import warnings
warnings.filterwarnings("ignore")
# setup arguments
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--lr_decay_step_size', type=int, default=50)
parser.add_argument('--opt_iters', type=int, default=10)
parser.add_argument('--eps', type=float, default=1.0)
parser.add_argument('--ratio', type=float, default=0.5)
parser.add_argument('--train', action='store_true')
parser.add_argument('--no_extra_mlp', action='store_true')
parser.add_argument('--multi_gpu', action='store_true', default=True)

args = parser.parse_args()

from pytorch_memlab import LineProfiler

def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)

def reshape_batch(example_graph):
    batch_size = args.batch_size
    example_graph.adj = example_graph.adj.unsqueeze(0)
    num_nodes = example_graph.adj.shape[2]
    example_graph.adj = torch.reshape(example_graph.adj, (batch_size, num_nodes, -1))
    example_graph.x = example_graph.x.unsqueeze(0)  
    embedding_shape = example_graph.x.shape[2]
    example_graph.x = torch.reshape(example_graph.x, (batch_size, num_nodes, -1))
    example_graph.y = example_graph.y.unsqueeze(0) 
    example_graph.y = torch.reshape(example_graph.y, (batch_size, -1))
    example_graph.mask = example_graph.mask.unsqueeze(0)
    example_graph.mask = torch.reshape(example_graph.mask, (batch_size, -1))
    return example_graph

def validation_test(model, dataset):
    print("Running validation")
    # make data loader
    #loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, exclude_keys=["label", "tfidf"], drop_last=True)
    loader = DataListLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    # go through the validation set
    total_loss = 0
    for example_graph in tqdm(loader):
        if not args.multi_gpu:
            example_graph = reshape_batch(example_graph)
        example_graph = example_graph.to(device)
        xs, new_adjs, Ss, opt_loss, output_indices = model(example_graph)
        if opt_loss == 0.0:
            continue
        total_loss += opt_loss.item() * num_graphs(example_graph)

    # average
    return total_loss / len(loader.dataset)

#@profile_every(1)
def train_iteration(model, optimizer, loader):
    print("iteration")
    model.train()
    total_loss = 0
    for example_graph in tqdm(loader):
        print(args.multi_gpu)
        if not args.multi_gpu:
            example_graph = reshape_batch(example_graph)
        optimizer.zero_grad()
        if not args.multi_gpu:
            example_graph = example_graph.to(device)
        # xs, new_adj, S, opt_loss = model(data, epsilon=0.01, opt_epochs=100)
        xs, new_adjs, Ss, opt_loss, output_indices = model(example_graph)
        if opt_loss == 0.0:
            continue
        opt_loss.backward()
        total_loss += opt_loss.item() * num_graphs(example_graph)
        optimizer.step()
    return total_loss / len(loader.dataset)

#@profile
def train(model, train_dataset, validation_dataset, save_dir="$GRAPH_SUM/src/ot_coarsening/model_cache"):
    dirpath = os.path.join(os.path.expandvars(save_dir),
                           "savedmodels_eps"+str(args.eps)+"_iter"+str(args.opt_iters))
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
    model_path = dirpath + "/opt_cnn_params.pkl"
    #unsupervised training
    #perm = torch.randperm(len(dataset))
    #train_id = int(0.8*len(dataset))
    #train_index = perm[:train_id]
    #val_index = perm[train_id:]
    # print("num_layers, hidden", num_layers, hidden)
    #train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, exclude_keys=["label", "tfidf"], drop_last=True, num_workers=2, multiprocessing_context='spawn')
    #train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, exclude_keys=["label", "tfidf"], drop_last=True)
    train_loader = DataListLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    if args.multi_gpu:
        model.module.to(device).reset_parameters()
    else:
        model.to(device).reset_parameters()
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
            "Rouge Validation Loss": rouge_validation_loss,
            "Unsupervised Validation Loss": unsupervised_validation_loss
        } 
        wandb.log(loss_dictionary)

def rouge_validation(model, dataset):
    rouge_evaluations = evaluation.perform_rouge_evaluations(model, dataset)

def init_model(dataset):
    num_layers = 1
    num_hiddens = 128
    model = Coarsening(dataset, num_hiddens, ratio=args.ratio, epsilon=args.eps, opt_epochs=args.opt_iters)
    # model = MultiLayerCoarsening(dataset, num_hiddens, ratio=args.ratio)
    if args.multi_gpu:
        model = DataParallel(model)
    
    return model

def setup_logging():
    """
        Sets up logging for a model training and evaluation pass 
        with weights and biases.
    """
    # 1. Start a new run
    wandb.init(project='graph-summarization', entity='helblazer811')
    # 2. Save model inputs and hyperparameters
    config = wandb.config
    # 3. Log gradients and model parameters
 
"""
    Main function for performing OTCoarsening on a DailyMail graph
"""
def main():
    # Setup logging
    setup_logging()
    # Make graph constructor
    graph_constructor = CNNDailyMailGraphConstructor()
    # Setup CNNDailyMail data
    train_dataset = CNNDailyMail(graph_constructor=graph_constructor, perform_processing=False, proportion_of_dataset=0.01)
    validation_dataset = CNNDailyMail(graph_constructor=graph_constructor, mode="val", perform_processing=False)
    # Initialize the model
    model = init_model(train_dataset)
    # Run training
    train(model, train_dataset, validation_dataset)

if __name__ == "__main__":
    main()
