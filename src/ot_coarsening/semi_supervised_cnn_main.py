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
import time
import numpy as np
import argparse
from tqdm import tqdm
from torch_geometric.data import DataListLoader, DataLoader, DenseDataLoader as DenseLoader, Batch
from torch_geometric.nn import DataParallel
from torch.optim import Adam
import warnings
warnings.filterwarnings("ignore")

# setup arguments
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser()
parser.add_argument('--unsupervised_epochs', type=int, default=20)
parser.add_argument('--supervised_epochs', type=int, default=0)
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
parser.add_argument('--similarity', type=bool, default=True)

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
    supervised_loss = 0
    unsupervised_loss = 0
    for example_graph in tqdm(loader):
        #example_graph = reshape_batch(example_graph)
        example_graph.to(device)
        supervised_loss, unsupervised_loss, loss, coarse_indices = model(example_graph, num_output_sentences=args.num_output_sentences)
        if loss == 0.0:
            continue
        total_loss += loss.item() * num_graphs(example_graph)
        supervised_loss += supervised_loss.item() * num_graphs(example_graph)
        unsupervised_loss += unsupervised_loss.item() * num_graphs(example_graph)
    # average
    return unsupervised_loss / len(loader.dataset), supervised_loss / len(loader.dataset)

def train_iteration(model, optimizer, loader, mode="unsupervised"):
    model.train()
    total_loss = 0
    supervised_loss = 0
    unsupervised_loss = 0
    for graph_index, example_graph in enumerate(tqdm(loader)):
        example_graph.to(device)
        optimizer.zero_grad()
        supervised_loss, unsupervised_loss, loss, coarse_indices = model(example_graph, num_output_sentences=args.num_output_sentences)
        if loss == 0.0:
            continue
        if mode == "both":
            loss.backward()
        elif mode == "unsupervised":
            unsupervised_loss.backward()
        elif mode == "supervised":
            supervised_loss.backward()
        total_loss += loss.item() * num_graphs(example_graph)
        supervised_loss += supervised_loss.item() * num_graphs(example_graph)
        unsupervised_loss += unsupervised_loss.item() * num_graphs(example_graph)
        optimizer.step()
    
    # empty reserved memory
    return unsupervised_loss / len(loader.dataset), supervised_loss / len(loader.dataset)

def train(model, unsupervised_train_dataset, supervised_train_dataset, validation_dataset, save_dir="$GRAPH_SUM/src/ot_coarsening/model_cache", run_name=None):
    if run_name is None:
        raise Exception("run_name is None in train")
    dirpath = os.path.join(os.path.expandvars(save_dir), run_name)
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
    model_path = os.path.join(dirpath, "savedmodels_eps"+str(args.eps)+"_iter"+str(args.opt_iters)+".pt")
    # setup loaders
    unsupervised_train_loader = DataLoader(unsupervised_train_dataset, batch_size=args.batch_size, shuffle=True, exclude_keys=["label", "tfidf"], drop_last=True)
    supervised_train_loader = DataLoader(supervised_train_dataset, batch_size=args.batch_size, shuffle=True, exclude_keys=["label", "tfidf"], drop_last=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    # go through each training iteration
    total_epochs = args.unsupervised_epochs + args.supervised_epochs
    model.to(device).reset_parameters()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=0.0001) # adding a negative weight regularizaiton such that it cannot be zero.
    mode = "unsupervised"
    for epoch in tqdm(range(1, total_epochs + 1)):
        if epoch > args.unsupervised_epochs:
            mode = "supervised" 
        # select the correct loaders
        if mode == "supervised":
            train_loader = supervised_train_loader
        else:
            train_loader = unsupervised_train_loader
        # Do a training iteration
        unsupervised_train_loss, supervised_train_loss = train_iteration(model, optimizer, train_loader, mode=mode)
        # Get the unsupervised validation loss
        unsupervised_validation_loss, supervised_validation_loss = validation_test(model, validation_dataset)
        # Get the Rouge train loss
        rouge_validation_loss = rouge_validation(model, validation_dataset)
        # Get the Rouge validation loss 
        # Log the loss
        loss_dictionary = {
            "Unsupervised Train Loss": unsupervised_train_loss,
            "Supervised Train Loss": supervised_train_loss,
            "Rouge Validation Loss": rouge_validation_loss,
            "Unsupervised Validation Loss": unsupervised_validation_loss,
            "Supervised Validation Loss": supervised_validation_loss
        } 
        wandb.log(loss_dictionary)
        # save the model
        torch.save(model, model_path)
        
def rouge_validation(model, dataset):
    rouge_evaluations = evaluation.perform_rouge_evaluations(model, dataset, dense=args.dense)

def init_model(dataset, model_type="ot_coarsening", dense=False):
    num_layers = 1
    num_hiddens = 2048
    in_channels = dataset.dimensionality
    out_channels = dataset.dimensionality
    model = GraphUNetCoarsening(in_channels, num_hiddens, out_channels, num_layers)
    
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
    # 3. Log the config
    # 4. Return the run name
    return wandb.run.name 
 
"""
    Main function for performing OTCoarsening on a DailyMail graph
"""
def main():
    torch.autograd.set_detect_anomaly(True)
    # Setup logging
    run_name = setup_logging()
    model_type = "u_net"
    # Setup CNNDailyMail data
    graph_constructor = CNNDailyMailGraphConstructor(similarity=args.similarity)
    supervised_train_dataset = CNNDailyMail(graph_constructor=graph_constructor, perform_processing=False, proportion_of_dataset=(0.0, 0.5), dense=args.dense)
    unsupervised_train_dataset = CNNDailyMail(graph_constructor=graph_constructor, perform_processing=False, proportion_of_dataset=(0.5, 1.0), dense=args.dense)
    validation_dataset = CNNDailyMail(graph_constructor=graph_constructor, mode="val", perform_processing=False, proportion_of_dataset=(0.0, 1.0), dense=args.dense)
    # Initialize the model
    model = init_model(unsupervised_train_dataset, dense=args.dense, model_type=model_type)
    # Run training
    train(model, unsupervised_train_dataset, supervised_train_dataset, validation_dataset, run_name=run_name)

if __name__ == "__main__":
    #torch.multiprocessing.set_start_method('spawn')# good solution !!!!("iteration")
    torch.backends.cudnn.benchmark = True
    main()
