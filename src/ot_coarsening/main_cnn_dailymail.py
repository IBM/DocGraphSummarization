import os
import sys
sys.path.append(os.environ["GRAPH_SUM"])
# from src.dataset_management.datasets import get_dataset
from src.ot_coarsening.dataset_management.cnn_daily_mail import CNNDailyMail
from src.ot_coarsening.dataset_management.graph_constructor import CNNDailyMailGraphConstructor
from src.ot_coarsening.ot_coarsening import MultiLayerCoarsening, Coarsening
import torch
import torch.nn.functional as F
import time
import numpy as np
import argparse
from tqdm import tqdm
from torch_geometric.data import DataLoader, DenseDataLoader as DenseLoader
from torch.optim import Adam

from pytorch_memlab import profile, set_target_gpu, profile_every
import warnings
warnings.filterwarnings("ignore")
# setup arguments
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)
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
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)

#@profile_every(1)
def train_iteration(model, optimizer, loader):
    print("iteration")
    model.train()
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        data = data.to(device)
        # xs, new_adj, S, opt_loss = model(data, epsilon=0.01, opt_epochs=100)
        xs, new_adjs, Ss, opt_loss = model(data, epsilon=args.eps, opt_epochs=args.opt_iters)
        if opt_loss == 0.0:
            continue
        opt_loss.backward()
        total_loss += opt_loss.item() * num_graphs(data)
        optimizer.step()
    return total_loss / len(loader.dataset)

#@profile
def train(model, dataset, save_dir="$GRAPH_SUM/src/ot_coarsening/model_cache"):
    dirpath = os.path.join(os.path.expandvars(save_dir),
                           "savedmodels_eps"+str(args.eps)+"_iter"+str(args.opt_iters))
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
    model_path = dirpath + "/opt_cnn_params.pkl"
    #unsupervised training
    perm = torch.randperm(len(dataset))
    train_id = int(0.8*len(dataset))
    train_index = perm[:train_id]
    val_index = perm[train_id:]
    # print("num_layers, hidden", num_layers, hidden)

    train_loader = DenseLoader(dataset[train_index], batch_size=args.batch_size, shuffle=True)
    model.to(device).reset_parameters()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=0.0001) # adding a negative weight regularizaiton such that it cannot be zero.

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t_start = time.perf_counter()
    val_losses = []
    val_loader = DenseLoader(dataset[val_index], batch_size=args.batch_size, shuffle=False)
    best_val_loss = 100000.0
    best_val_epoch = 0
    for epoch in tqdm(range(1, args.epochs + 1)):
        train_loss = train_iteration(model, optimizer, train_loader)
        print(train_loss)
        """
        # val_loss, val_x, val_adjs, val_Ss = eval_loss(model, val_loader)

        # val_losses.append(val_loss)
        eval_info = {
            'epoch': epoch,
            'train_loss': train_loss,
        #   'val_loss': val_losses[-1],
        }
        print(eval_info)
        if val_loss < best_val_loss:
            torch.save(model.state_dict(), model_path)
        #    best_val_loss = val_loss
            best_val_epoch = epoch
        if epoch-best_val_epoch>30:
            break
        """

def test(model, dataset):
    pass

def eval_loss(model, loader):
    pass

def init_model(dataset):
    num_layers = 1
    num_hiddens = 64
    model = Coarsening(dataset, num_hiddens, ratio=args.ratio)
    # model = MultiLayerCoarsening(dataset, num_hiddens, ratio=args.ratio)

    return model

"""
    Main function for performing OTCoarsening on a DailyMail graph
"""
def main():
    # Make graph constructor
    graph_constructor = CNNDailyMailGraphConstructor()
    # Setup CNNDailyMail data
    dataset = CNNDailyMail(graph_constructor=graph_constructor, perform_processing=False)
    # Initialize the model
    model = init_model(dataset)
    # Run training
    train(model, dataset)
    # Run evaluation
    test(model, dataset)

if __name__ == "__main__":
    main()
