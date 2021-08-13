import os
import json
import sys
sys.path.append(os.environ["GRAPH_SUM"])
from datasets import load_dataset
import nltk
nltk.download('punkt')
import src.ot_coarsening.dataset_management.tfidf_calculator as tfidf_calculator
import src.ot_coarsening.evaluation.rouge_evaluation as rouge_evaluation
import numpy as np
from operator import itemgetter
import jsonlines
from tqdm import tqdm

"""
    Uses huggingface to download cnndm
"""
def download_cnndm():
    dataset = load_dataset('cnn_dailymail', '3.0.0')
    return dataset

def filter_tokenized(tokenized):
    filtered_tokens = []
    for sent in tokenized:
        if sent == "." or sent == ".." or sent == "..." or sent == "":
            continue
        else:
            filtered_tokens.append(sent)
    return filtered_tokens

def load_dataset_labels(dataset, dataset_type="train"):
    # take the dataset
    if dataset_type == "val":
        dataset_type = "validation"
    dataset = dataset[dataset_type]
    # go through each document
    dataset_list = []
    index = 0
    for item in dataset:
        if index > 50000:
            break
        article = item["article"].lower()
        highlights = item["highlights"].lower()
        # split the items
        article_split = nltk.tokenize.sent_tokenize(article)
        article_split = filter_tokenized(article_split)
        highlight_split = nltk.tokenize.sent_tokenize(highlights)
        highlight_split = filter_tokenized(highlight_split)
        # make dict object
        dictionary = {"text": article_split, "summary":highlight_split}
        dataset_list.append(dictionary)
        index += 1

    return dataset_list

"""
   max rouge scoring sentence for each summary sentence
"""
def calculate_max_per_sentence_ranking(label_dict):
    sentences = label_dict["text"]
    summary = label_dict["summary"]
    num_summary_sentences = len(summary)
    # get each document sentences overlap score with each sentence in the summary
    sentence_rouge_scores = [] 
    for sentence in sentences:
        individual_sentence_scores = []
        for summary_sentence in summary:
            score = rouge_evaluation.compute_sentence_average_rouge_overlap(sentence, summary_sentence)
            individual_sentence_scores.append(score)
        sentence_rouge_scores.append(individual_sentence_scores)
    # convert to numpy
    # has a (document_sentence, summary_sentence)
    sentence_rouge_scores = np.array(sentence_rouge_scores)
    # argsort along document axis
    # this outputs a (document_sentence, summary_sentence) shape array where each doucment_sentence has a 
    # list of indices of how its score ranks for a specific summary sentence
    # lower is better because reversed
    argsorted_sentence_rouge_scores = np.argsort(sentence_rouge_scores, axis=0)[::-1]
    # go through each sentence and add its index to a list
    sentence_index_list = [[] for i in range(len(sentences))]
    for sentence_index in range(len(sentences)):
        summary_ranks = argsorted_sentence_rouge_scores[sentence_index]
        for summary_rank in summary_ranks:
            sentence_index_list[summary_rank].append(sentence_index)
    # sort each of the sentence_index_list individualy
    sorted_sentence_index_list = []
    for index, summary_ranks in enumerate(sentence_index_list):
        summary_ranks = np.sort(summary_ranks)
        # prepend the list with the index
        summary_ranks = np.concatenate((np.array([index]), summary_ranks))
        # convert back to list
        summary_ranks = summary_ranks.tolist()
        sorted_sentence_index_list.append(summary_ranks)
    # sort by the first value of sentence_index_list, if they are the same then sort by the second
    for summary_sentence_index in range(num_summary_sentences):
        # add one because the zeroth index is the index
        sorted_sentence_index_list.sort(key=itemgetter(summary_sentence_index + 1))
    # convert to np array
    sorted_sentence_index_list = np.array(sorted_sentence_index_list)    
    ranking = sorted_sentence_index_list[:, 0]
    ranking = sorted_sentence_index_list[:, 0].squeeze().tolist()
    
    return ranking

def calculate_max_per_sentence_rankings(label_dicts):
    for label_dict in label_dicts:
        rankings = calculate_max_per_sentence_ranking(label_dict)
        label_dict["max_per_rankings"] = rankings

def calculate_max_per_sentence_top3(label_dicts):
    for label_dict in label_dicts:
        rankings = label_dict["max_per_rankings"]
        if isinstance(rankings, int):
            rankings = [rankings]
        top_num = 3 if len(rankings) > 3 else len(rankings)
        label_dict["max_per_top3"] = rankings[0:top_num]

"""
   each sentence in the text ranked by its average overlap
"""
def calculate_average_score_rankings(label_dicts):
    for label_dict in tqdm(label_dicts):
        sentences = label_dict["text"]
        summary = label_dict["summary"]
        sentence_rouge = []
        for sentence in sentences:
            score = rouge_evaluation.compute_paragraph_average_rouge_overlap(sentence, summary)
            sentence_rouge.append(score)
        # get the argsort of sentence_rogue 
        ranking_index = np.argsort(sentence_rouge)[::-1]
        ranking_index = ranking_index.tolist()
        # save raning in object
        label_dict["average_rankings"] = ranking_index

def calculate_average_score_top3(label_dicts):
    for label_dict in label_dicts:
        label_dict["average_top3"] = label_dict["average_rankings"][0:3]
        label_dict["label"] = label_dict["average_top3"]

def load_dataset_tfidf(labels):
    tfidfs = []
    for label in tqdm(labels):
        # get sentence tfidf
        tfidf = tfidf_calculator.compute_sentence_tfidf(None, label)
        tfidfs.append(tfidf)
    return tfidfs

if __name__ == "__main__":
    # download the data
    dataset = download_cnndm()
    # load each datset type
    # dataset_types = ["train", "test", "val"]
    dataset_types = ["val"]
    for dataset_type in dataset_types:
        print("Loading {}".format(dataset_type))
        print("load dataset labels")
        label_dicts = load_dataset_labels(dataset, dataset_type=dataset_type)
        # calculate the average rouge rankings and modify the label dicts
        print("calculate average rankings")
        calculate_average_score_rankings(label_dicts)
        print(label_dicts)
        # calculate the average rouge top3
        calculate_average_score_top3(label_dicts)
        # calculate the max_rouge rankings
        print("calculate max per sentence rankings")
        calculate_max_per_sentence_rankings(label_dicts)
        # calculate the max rouge top3
        calculate_max_per_sentence_top3(label_dicts)
        # load tfidf
        print("calculate tfidf")
        tfidfs = load_dataset_tfidf(label_dicts)
        # save the tfidf and labels
        print("saving")
        with open(dataset_type+'.w2s.tfidf.jsonl', 'w') as outfile:
            for entry in tfidfs:
                json.dump(entry, outfile)
                outfile.write('\n')
        with open(dataset_type+'.label.jsonl', 'w') as outfile:
            for entry in label_dicts:
                print(entry)
                json.dump(entry, outfile)
                outfile.write('\n')
