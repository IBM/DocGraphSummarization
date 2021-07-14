import os
import sys
sys.path.append(os.environ["GRAPH_SUM"])
# from src.dataset_management.datasets import get_dataset
from src.ot_coarsening.dataset_management.duc.duc import DUC
from src.ot_coarsening.dataset_management.graph_constructor import DUCGraphConstructor
import torch
import torch.nn.functional as F
import time
import numpy as np
import argparse
from torch_geometric.data import DataLoader, DenseDataLoader as DenseLoader
from torch.optim import Adam
import warnings
warnings.filterwarnings("ignore")
# setup arguments
device = torch.device('cpu') #torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
    Main function for processing the dataset
"""
def main():
    # Make graph constructor
    graph_constructor = DUCGraphConstructor()
    # Setup CNNDailyMail data
    mode = "train"
    proportion_of_dataset = 0.1
    mode = "val"
    year = "2005"
    proportion_of_dataset = 0.1
    dataset = DUC(graph_constructor=graph_constructor, perform_processing=True, mode=mode, proportion_of_dataset=proportion_of_dataset, year=year)

if __name__ == "__main__":
    main()
