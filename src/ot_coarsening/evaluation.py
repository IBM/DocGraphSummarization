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
from src.ot_coarsening.dataset_management.graph_constructor import CNNDailyMailGraphConstructor
from src.ot_coarsening.dataset_management.cnn_daily_mail import CNNDailyMail
import collections
import wandb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    rouge = Rouge() 
    hyps = []
    refs = []
    for i in range(len(predicted_summaries)):
        predicted_summary = predicted_summaries[i]
        true_summary = true_summaries[i]
        hyp = "\n".join(predicted_summary)
        ref = "\n".join(true_summary)
        hyps.append(hyp)
        refs.append(ref)
    
    score = rouge.get_scores(hyps, refs, avg=True)
    return score

def get_coarse_sentences(ground_truth, coarse_indices, num_sentences, output_node_count):
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
    text = ground_truth["text"]
    coarse_sentences = [text[sentence_index] for sentence_index in sentence_indices]
    return coarse_sentences, sentence_indices

def generate_predicted_summary(model, example_graph, ground_truth, output_node_count):
    """
        Generates a textual summmary of the example_graph using
        the given model.
    """
    # put model in eval mode
    model.eval()
    # run forward pass using model
    num_sentences = len(ground_truth["text"])
    example_graph.to(device)
    example_graph = Batch.from_data_list([example_graph])
    #example_graph.x = example_graph.x.unsqueeze(0)  
    #example_graph.y = example_graph.y.unsqueeze(0) 
    #example_graph.mask = example_graph.mask.unsqueeze(0)
    #example_graph.adj = example_graph.adj.unsqueeze(0)
    output_node_counts = torch.Tensor([output_node_count])
    _, _, _, _, _, coarse_indices = model.forward(example_graph, output_node_counts=output_node_counts, num_sentences=[num_sentences]) 
    # should be a 1D tensor of indices
    coarse_indices = coarse_indices[0]
    predicted_summary, sentence_indices = get_coarse_sentences(ground_truth, coarse_indices, num_sentences, output_node_count)
    assert len(predicted_summary) == output_node_count
    
    return predicted_summary, sentence_indices

def compute_label_accuracy(predicted_labels, true_labels):
    num_correct = 0.0
    for predicted in predicted_labels:
        if predicted in true_labels:
            num_correct += 1
    
    return num_correct / len(true_labels)

def save_histogram(saved_sentence_inds, file_path):
    plt.clf()
    plt.hist(saved_sentence_inds, bins=20)  # density=False would make counts
    plt.xlim(0, 1)
    plt.show()
    plt.savefig(file_path)
            
def perform_rouge_evaluations(model, dataset, serialize=True, log=True):
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
        output_node_count = 3# len(example_label)
        # get summary prediction
        predicted_summary, sentence_indices = generate_predicted_summary(model, example_graph, ground_truth, output_node_count)
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

def perform_random_rouge_baseline(dataset):
    predicted_summaries = []
    true_summaries = []
    label_accuracies = []
    saved_sentence_inds = [] 
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
        output_node_count = 3 #len(true_summary)
        # choose random summary 
        predicted_summary_indices = random.sample(range(0, len(ground_truth["text"])), output_node_count)
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

if __name__ == "__main__":
    # get a random baseline 
    # run valuation on a saved model
    graph_constructor = CNNDailyMailGraphConstructor()
    validation_dataset = CNNDailyMail(graph_constructor=graph_constructor, mode="val", perform_processing=False)
    print("Random evaluation baseline")
    perform_random_rouge_baseline(validation_dataset)
    print("Evaluate existing model")
    # deserialize a model
    model_path = "model_cache/snowy-tree-534/savedmodels_eps1.0_iter10.pt"
    model = torch.load(model_path)
    perform_rouge_evaluations(model, validation_dataset, serialize=False, log=False)
