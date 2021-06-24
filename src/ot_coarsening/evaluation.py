#!/usr/bin/env python3
import jsonlines
import pandas as pd
from rouge import Rouge
import os
import numpy as np
from tqdm import tqdm
import torch
from torch_geometric.data import Batch, Data
import collections
import wandb

def load_labels(dataset_path):
    """
        Assumes that the labels have the same structure as the 
        CNNDailyMailDataset
    """
    # WARNING THIS TAKES ALOT OF MEMORY
    dataset_list = []
    # load the file with jsonlines
    with jsonlines.open(dataset_path) as label_reader:
        for index, label in enumerate(tqdm(label_reader)):
            # check number of nodes
            dataset_list.append(label)
 
    # return the data
    return dataset_list

def average_rouge(rouge_list):
    rouge_types = list(rouge_list[0].keys())
    sub_scores = list(rouge_list[0][rouge_types[0]].keys())
    length_of_rouge_list = len(rouge_list)
    average_rouge_dict = {rouge_type : {} for rouge_type in rouge_types}
    for rouge_type in rouge_types:
        for sub_score in sub_scores:
            total = sum(rouge_dict[rouge_type][sub_score] for rouge_dict in rouge_list)
            average = total / length_of_rouge_list
            average_rouge_dict[rouge_type][sub_score] = average

    return average_rouge_dict

def log_rouge_dictionary(rouge_dictionary):

    def flatten(d, parent_key='', sep='_'):
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, collections.MutableMapping):
                items.extend(flatten(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
        
    flattened = flatten(rouge_dictionary)
    wandb.log(flattened)

def compute_rouge(predicted_summaries, true_summaries):
    """
        Performs a single set of rouge evaluations
    """
    rouge = Rouge()
    rouge_scores = []
    for index in range(len(predicted_summaries)):
        predicted_summary = predicted_summaries[index]
        true_summary = true_summaries[index]
        rouge_score = rouge.get_scores(predicted_summary, true_summary, avg=True)
        rouge_scores.append(rouge_score)    
    average_rouge_scores = average_rouge(rouge_scores)
        
    return average_rouge_scores

def get_coarse_sentences(example_label, coarse_indices, num_sentences, output_node_count):
    """
        Returns the coarse sentences in an example graph. 
    """
    # get coarse sentence indices
    coarse_indices = coarse_indices.cpu().numpy()
    sentence_indices = np.where(coarse_indices < num_sentences)[0].squeeze()
    sentence_indices = coarse_indices[sentence_indices]
    if len(np.shape(sentence_indices)) == 0:
        sentence_indices = [sentence_indices]
    if np.shape(sentence_indices)[0] > output_node_count:
        sentence_indices = sentence_indices[0: output_node_count]
    # get the sentences
    text = example_label["text"]
    coarse_sentences = [text[sentence_index] for sentence_index in sentence_indices]
    return coarse_sentences 

def generate_predicted_summary(model, example_graph, example_label, output_node_count):
    """
        Generates a textual summmary of the example_graph using
        the given model.
    """
    # put model in eval mode
    model.eval()
    # run forward pass using model
    num_sentences = len(example_label["text"])
    example_graph = Batch.from_data_list([example_graph])
    example_graph.x = example_graph.x.unsqueeze(0)  
    example_graph.y = example_graph.y.unsqueeze(0) 
    example_graph.mask = example_graph.mask.unsqueeze(0)
    example_graph.adj = example_graph.adj.unsqueeze(0)
    output_node_counts = torch.Tensor([output_node_count])
    _, _, _, _, coarse_indices = model.forward(example_graph, output_node_counts=output_node_counts, num_sentences=num_sentences) 
    # should be a 1D tensor of indices
    coarse_indices = coarse_indices[0]
    predicted_summary = get_coarse_sentences(example_label, coarse_indices, num_sentences, output_node_count)
    assert len(predicted_summary) == output_node_count
    
    return predicted_summary

def perform_rouge_evaluations(model, dataset, serialize=True):
    """
        Perform rouge-1, rouge-2, and rouge-l evaluation given a
        trained OTCoarsening model on the given evaluation dataset.
        
        Returns
            - Average rouge-1, rouge-2, and rouge-l scores on the 
            validation dataset and returns it as a dictionary.
    """
    model.eval()
    # load the labels
    # labels = load_labels(os.path.join(dataset.root, dataset.label_path))
    # go though each example in the dataset
    predicted_summaries = []
    true_summaries = []
    for example_index in range(len(dataset)):
        # get the example graph
        example_graph = dataset[example_index]
        # get the labels
        # example_label = labels[example_index]
        example_label = example_graph.label
        true_summary = example_label["summary"]
        if len(true_summary) > len(example_label["text"]):
            continue
        true_summaries.append(true_summary)
        output_node_count = len(true_summary)
        # get summary prediction
        predicted_summary = generate_predicted_summary(model, example_graph, example_label, output_node_count)
        predicted_summaries.append(predicted_summary)
    
    rouge_dictionary = compute_rouge(predicted_summaries, true_summaries)
    # log rouge
    log_rouge_dictionary(rouge_dictionary)
    # serialize
    if serialize:
        pass

    return rouge_dictionary

