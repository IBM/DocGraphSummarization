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
import collections
import wandb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def save_histogram(saved_sentence_inds, file_path):
    plt.clf()
    plt.hist(saved_sentence_inds, bins=20)  # density=False would make counts
    plt.xlim(0, 1)
    plt.show()
    plt.savefig(file_path)

def load_labels(dataset_path):
    """
        Assumes that the labels have the same structure as the 
        CNNDailyMailDataset
    """
    # WARNING THIS TAKES ALOT OF MEMORY
    dataset_list = []
    # load the file with jsonlines
    with jsonlines.open(dataset_path) as label_reader:
        for index, ground_truth in enumerate(tqdm(label_reader)):
            # check number of nodes
            dataset_list.append(ground_truth)
 
    # return the data
    return dataset_list

def compute_rouge(predicted_summaries, true_summaries):
    rouge = Rouge() 
    hyps = []
    refs = []
    for i in range(len(predicted_summaries)):
        print(i)
        predicted_summary = predicted_summaries[i]
        true_summary = true_summaries[i]
        hyp = "\n".join(predicted_summary)
        ref = "\n".join(true_summary)
        hyps.append(hyp)
        refs.append(ref)
     
    score = rouge.get_scores(hyps, refs, avg=True)
    return score

def get_coarse_sentences(ground_truth, coarse_indices):
    """
        Returns the coarse sentences in an example graph. 
    """
    # get coarse sentence indices
    coarse_indices = coarse_indices.cpu().numpy()
    num_sentences = len(ground_truth["text"]) 
    sentence_indices = np.where(coarse_indices < num_sentences)[0].squeeze()
    sentence_indices = coarse_indices[sentence_indices]
    if len(np.shape(sentence_indices)) == 0:
        sentence_indices = [sentence_indices]
    # get the sentences
    text = ground_truth["text"]
    coarse_sentences = [text[sentence_index] for sentence_index in sentence_indices]
    return coarse_sentences, sentence_indices

"""
    Generates a textual summmary of the example_graph using
    the given model.
"""
def generate_predicted_summary(model, example_graph, ground_truth, num_output_sentences, dense=False):
    # put model in eval mode
    model.eval()
    # run forward pass using model
    num_sentences = len(ground_truth["text"])
    example_graph.to(device)
    example_graph = Batch.from_data_list([example_graph])
    if model.__class__.__name__ == "GraphUNetCoarsening":
       loss_dict, loss, coarse_indices = model(example_graph, num_output_sentences=num_output_sentences)
    else:
        _, _, _, _, _, coarse_indices = model(example_graph, num_output_sentences=num_output_sentences)
    # should be a 1D tensor of indices
    if isinstance(coarse_indices, list):
        coarse_indices = coarse_indices[0]
    predicted_summary, sentence_indices = get_coarse_sentences(ground_truth, coarse_indices)
    
    return predicted_summary, sentence_indices

def compute_label_accuracy(predicted_labels, true_labels):
    num_correct = 0.0
    for predicted in predicted_labels:
        if predicted in true_labels:
            num_correct += 1
    
    return num_correct / len(true_labels)
           
def perform_rouge_evaluations(model, dataset, serialize=True, log=True, num_output_sentences=3, dense=False):
    """
        Perform rouge-1, rouge-2, and rouge-l evaluation given a
        trained OTCoarsening model on the given evaluation dataset.
        
        Returns
            - Average rouge-1, rouge-2, and rouge-l scores on the 
            validation dataset and returns it as a dictionary.
    """
    model.eval()
    # load the labebels
    # labels = load_labels(os.path.join(dataset.root, dataset.label_path))
    # go though each example in the dataset
    predicted_summaries = []
    true_summaries = []
    label_accuracies = []
    saved_sentence_inds = []
    ground_truth_labels = []
    for example_index in range(len(dataset)):
        # get the example graph
        example_graph = dataset[example_index]
        # get the labels
        # example_label = labels[example_index]
        ground_truth = example_graph.label
        example_label = ground_truth["label"]
        ground_truth_labels.append(np.array(example_label) / len(ground_truth["text"]))
        true_summary = ground_truth["summary"]
        if len(true_summary) > len(ground_truth["text"]) or len(example_label) == 0:
            continue
        true_summaries.append(true_summary)
        # get summary prediction
        predicted_summary, sentence_indices = generate_predicted_summary(model, example_graph, ground_truth, num_output_sentences, dense=False)
        saved_sentence_inds.append(np.array(sentence_indices) / len(ground_truth["text"]))
        label_accuracy = compute_label_accuracy(sentence_indices, example_label)
        label_accuracies.append(label_accuracy)
        predicted_summaries.append(predicted_summary)

    save_histogram(np.array(saved_sentence_inds).flatten(), "not_random.png")
    save_histogram(np.array(ground_truth_labels).flatten(), "ground_truth_labels.png")
    
    rouge_dictionary = compute_rouge(predicted_summaries, true_summaries)
    # log rouge
    if log:
        log_rouge_dictionary(rouge_dictionary)
    print(rouge_dictionary)
    # mean label accuracy
    label_accuracy = np.mean(label_accuracies)
    if log:
        wandb.log({"label_accuracy": label_accuracy})
    print(label_accuracy)
    # serialize
    if serialize:
        pass

    return rouge_dictionary

def perform_pregenerated_summary(true_summaries, predicted_summaries, true_labels, predicted_labels, num_output_sentences=3):
    label_accuracies = []
    assert len(true_summaries) == len(predicted_summaries)
    for example_index in range(len(true_summaries)):
        true_summary = true_summaries[example_index]
        predicted_summary = predicted_summaries[example_index]
        true_label = true_labels[example_index]
        predicted_label = predicted_labels[example_index]
        label_accuracy = compute_label_accuracy(predicted_label, true_label)
        label_accuracies.append(label_accuracy)
    label_accuracy = np.mean(label_accuracies)
    print("Label accuracy")
    print(label_accuracy)
    rouge_dictionary = compute_rouge(predicted_summaries, true_summaries)
    print(rouge_dictionary)
    return rouge_dictionary

def perform_top_3_baseline(dataset):
    predicted_summaries = []
    true_summaries = []
    label_accuracies = []
    saved_sentence_inds = [] 
    num_output_sentences = dataset.num_output_sentences
    for example_index in range(len(dataset)):
        # get the example graph
        example_graph = dataset[example_index]
        # get the labels
        # example_label = labels[example_index]
        ground_truth = example_graph.label
        example_label  = ground_truth["label"]
        true_summary = ground_truth["summary"]
        if len(true_summary) > len(ground_truth["text"]) or len(example_label) == 0:
            continue
        true_summaries.append(true_summary)
        # choose random summary 
        predicted_summary_indices = [0, 1, 2]
        saved_sentence_inds.append(np.array(predicted_summary_indices) / len(ground_truth["text"]))
        label_accuracy = compute_label_accuracy(predicted_summary_indices, example_label)
        label_accuracies.append(label_accuracy)
        predicted_summary = [ground_truth["text"][i] for i in predicted_summary_indices]
        predicted_summaries.append(predicted_summary)

    save_histogram(np.array(saved_sentence_inds).flatten(), "random.png")
    
    label_accuracy = np.mean(label_accuracies)
    print("Label accuracy")
    print(label_accuracy)
    rouge_dictionary = compute_rouge(predicted_summaries, true_summaries)
    print(rouge_dictionary)
    return rouge_dictionary



def perform_random_rouge_baseline(dataset):
    predicted_summaries = []
    true_summaries = []
    label_accuracies = []
    saved_sentence_inds = [] 
    num_output_sentences = dataset.num_output_sentences
    for example_index in range(len(dataset)):
        # get the example graph
        example_graph = dataset[example_index]
        # get the labels
        # example_label = labels[example_index]
        ground_truth = example_graph.label
        example_label  = ground_truth["label"]
        true_summary = ground_truth["summary"]
        if len(true_summary) > len(ground_truth["text"]) or len(example_label) == 0:
            continue
        true_summaries.append(true_summary)
        # choose random summary 
        predicted_summary_indices = random.sample(range(0, len(ground_truth["text"])), num_output_sentences)
        saved_sentence_inds.append(np.array(predicted_summary_indices) / len(ground_truth["text"]))
        label_accuracy = compute_label_accuracy(predicted_summary_indices, example_label)
        label_accuracies.append(label_accuracy)
        predicted_summary = [ground_truth["text"][i] for i in predicted_summary_indices]
        predicted_summaries.append(predicted_summary)

    save_histogram(np.array(saved_sentence_inds).flatten(), "random.png")
    
    label_accuracy = np.mean(label_accuracies)
    print("Label accuracy")
    print(label_accuracy)
    rouge_dictionary = compute_rouge(predicted_summaries, true_summaries)
    print(rouge_dictionary)
    return rouge_dictionary

"""
    Compute the average f score from the rouge dict
"""
def compute_average_rouge_f_statistic(rouge_dict):
    score_types = ["rouge-1", "rouge-2", "rouge-l"]
    average_f = sum([rouge_dict[score_type]["f"] for score_type in score_types]) / len(score_types)
    return average_f

def compute_paragraph_average_rouge_overlap(sentence, summary):
    rouge = Rouge() 
    copied_sentences = [sentence for i in range(len(summary))]
    print(copied_sentences)
    score = rouge.get_scores(copied_sentences, summary, avg=True)
    average_f = compute_average_rouge_f_statistic(score)
    return average_f

def compute_sentence_average_rouge_overlap(sentence_one, sentence_two):
    rouge = Rouge() 
    score = rouge.get_scores([sentence_one], [sentence_two], avg=True)
    average_f = compute_average_rouge_f_statistic(score)
    return average_f
