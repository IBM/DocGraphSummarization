#!/usr/bin/env python3
import jsonlines
import pandas as pd
from rouge import Rouge
import os
import numpy as np
from tqdm import tqdm
import random
import torch
from torch_geometric.data import DataLoader, Batch, Data
import matplotlib.pyplot as plt
import sys
sys.path.append(os.environ["GRAPH_SUM"])
from src.ot_coarsening.dataset_management.cnndm.graph_constructor import CNNDailyMailGraphConstructor
from src.ot_coarsening.dataset_management.cnndm.cnn_daily_mail import CNNDailyMail
import src.ot_coarsening.evaluation.rouge_evaluation as rouge_evaluation
import collections

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    # get a random baseline 
    # run valuation on a saved model
    graph_constructor = CNNDailyMailGraphConstructor()
    validation_dataset = CNNDailyMail(graph_constructor=graph_constructor, mode="val", perform_processing=False, highlights=True)
    print("Random baseline")
    rouge_evaluation.perform_random_rouge_baseline(validation_dataset)
    print("Top3 baseline")
    rouge_evaluation.perform_top_3_baseline(validation_dataset)
    # deserialize a model
    #model_path = "model_cache/snowy-tree-534/savedmodels_eps1.0_iter10.pt"
    #model = torch.load(model_path)
    #rouge_evaluation.perform_rouge_evaluations(model, validation_dataset, serialize=False, log=False)
