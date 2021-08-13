import os
import sys
sys.path.append(os.environ["GRAPH_SUM"])
# from src.dataset_management.datasets import get_dataset
from src.ot_coarsening.dataset_management.cnndm.cnn_daily_mail import CNNDailyMail
from src.ot_coarsening.dataset_management.cnndm.graph_constructor import CNNDailyMailGraphConstructor
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
device = torch.device('cuda') #torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
    Main function for processing the dataset
"""
def main():
    # Make graph constructor
    graph_constructor = CNNDailyMailGraphConstructor(similarity=False, prepend_ordering=True)
    # Setup CNNDailyMail data
    mode = "train"
    proportion_of_dataset = 1.0
    dataset = CNNDailyMail(graph_constructor=graph_constructor, perform_processing=True, mode=mode, proportion_of_dataset=(0, proportion_of_dataset), highlights=True, overwrite_existing=False)
    mode = "val"
    proportion_of_dataset = 1.0
    #dataset = CNNDailyMail(graph_constructor=graph_constructor, perform_processing=True, mode=mode, proportion_of_dataset=(0, proportion_of_dataset), highlights=True, overwrite_existing=False)

if __name__ == "__main__":
    main()
