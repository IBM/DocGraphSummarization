import os
import gc
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
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--lr_decay_step_size', type=int, default=50)
parser.add_argument('--opt_iters', type=int, default=10)
parser.add_argument('--eps', type=float, default=1.0)
parser.add_argument('--ratio', type=float, default=0.5)
parser.add_argument('--train', action='store_true')
parser.add_argument('--no_extra_mlp', action='store_true')

args = parser.parse_args()

from pytorch_memlab import LineProfiler

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
        #example_graph = reshape_batch(example_graph)
        example_graph = example_graph.to(device)
        xs, edge_index, edge_attr, Ss, opt_loss, output_indices = model(example_graph)
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
        #print_resident_tensors()
        #example_graph = reshape_batch(example_graph)
        optimizer.zero_grad()
        # xs, new_adj, S, opt_loss = model(data, epsilon=0.01, opt_epochs=100)
        with torch.cuda.amp.autocast():
            xs, edge_index, edge_attr, Ss, opt_loss, output_indices = model(example_graph)
        #print("opt loss")
        #print(opt_loss)
        if opt_loss == 0.0:
            continue
        opt_loss.backward()
        total_loss += opt_loss.item() * num_graphs(example_graph)
        optimizer.step()

    # empty reserved memory
    #torch.cuda.empty_cache()
    return total_loss / len(loader.dataset)

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
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, exclude_keys=["label", "tfidf"], drop_last=True)#, num_workers=2)
    model.to(device).reset_parameters()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=0.0001) # adding a negative weight regularizaiton such that it cannot be zero.

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    for epoch in tqdm(range(1, args.epochs + 1)):
        # Do a training iteration
        train_loss = train_iteration(model, optimizer, train_loader)
        # Get the unsupervised validation loss
        #unsupervised_validation_loss = validation_test(model, validation_dataset)
        unsupervised_validation_loss = 0.0
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
    num_hiddens = 64
    model = Coarsening(dataset, num_hiddens, ratio=args.ratio, epsilon=args.eps, opt_epochs=args.opt_iters)
    # model = MultiLayerCoarsening(dataset, num_hiddens, ratio=args.ratio)
    
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
 
def print_resident_tensors():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
        except:
            pass 

"""
    Main function for performing OTCoarsening on a DailyMail graph
"""
def main():
    # Setup logging
    setup_logging()
    # Make graph constructor
    graph_constructor = CNNDailyMailGraphConstructor()
    # Setup CNNDailyMail data
    train_dataset = CNNDailyMail(graph_constructor=graph_constructor, perform_processing=False)
    validation_dataset = CNNDailyMail(graph_constructor=graph_constructor, mode="val", perform_processing=False)
    # Initialize the model
    model = init_model(train_dataset)
    # Run training
    train(model, train_dataset, validation_dataset)

if __name__ == "__main__":
    #torch.multiprocessing.set_start_method('spawn')# good solution !!!!("iteration")
    torch.backends.cudnn.benchmark = True
    main()
