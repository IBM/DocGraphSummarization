import os
import sys
sys.path.append(os.environ["GRAPH_SUM"])
# from src.dataset_management.datasets import get_dataset
from src.ot_coarsening.dataset_management.cnn_daily_mail import CNNDailyMail
from src.ot_coarsening.dataset_management.graph_constructor import CNNDailyMailGraphConstructor
from src.ot_coarsening.ot_coarsening import MultiLayerCoarsening
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
    Main function for processing the dataset
"""
def main():
    # Make graph constructor
    graph_constructor = CNNDailyMailGraphConstructor()
    # Setup CNNDailyMail data
    #mode = "val"
    #proportion_of_dataset = 0.10
    #dataset = CNNDailyMail(graph_constructor=graph_constructor, perform_processing=True, mode=mode, proportion_of_dataset=proportion_of_dataset)
    mode = "train"
    proportion_of_dataset = 0.10
    dataset = CNNDailyMail(graph_constructor=graph_constructor, perform_processing=True, mode=mode, proportion_of_dataset=proportion_of_dataset)

if __name__ == "__main__":
    main()
