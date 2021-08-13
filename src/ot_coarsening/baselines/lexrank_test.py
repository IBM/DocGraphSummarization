from lexrank import STOPWORDS, LexRank
import numpy as np
from path import Path
import sys
import os
import json
sys.path.append(os.environ["GRAPH_SUM"])
from src.ot_coarsening.dataset_management.cnndm.cnn_daily_mail import CNNDailyMail
from src.ot_coarsening.evaluation.rouge_evaluation import perform_pregenerated_summary

def import_json_dataset(doc_type="train"):
    # load up the json file
    json_label_path = f"/dccstor/helbling1/data/CNNDM/highlight_unprocessed/{doc_type}.label.jsonl"
    document_text = []
    true_summaries = []
    true_labels = []
    num_test = 10000
    total = 0
    with open(json_label_path, "r") as json_file:
        for line in json_file:
            if total > num_test:
                break
            total += 1
            json_object = json.loads(line)
            text =  json_object["text"]
            summary = json_object["summary"]
            label = json_object["label"]
            document_text.append(text)
            true_summaries.append(summary)
            true_labels.append(label)
    return document_text, true_summaries, true_labels

def run_rouge_evaluation(true_output, predicted_output, true_labels, predicted_labels, num_output_sentences=3):
    perform_pregenerated_summary(true_output, predicted_output, true_labels, predicted_labels, num_output_sentences=num_output_sentences)

def run_lexrank_baseline(num_train_examples=-1, num_test_examples=-1, num_output_sentences=3):
    # setup the dataset
    train_document_text, train_true_summary, train_true_labels = import_json_dataset(doc_type="train")
    train_document_text = train_document_text[0:num_train_examples]
    test_document_text, test_true_summary, test_true_labels = import_json_dataset(doc_type="test")
    test_document_text = test_document_text[0:num_test_examples]
    test_true_summary = test_true_summary[0:num_test_examples]
    # make return arrays
    predicted_labels = [] 
    predicted_outputs = []
    # perform lexrank
    lxr = LexRank(train_document_text, stopwords=STOPWORDS['en'])
    for example_index in range(len(test_document_text)):
        test_document = test_document_text[example_index]
        # rank the sentences
        lex_scores = lxr.rank_sentences(
            test_document,
            threshold=None,
            fast_power_method=False,
        )
        # get argsort of scores
        num_out_sents = num_output_sentences if num_output_sentences <= len(test_document) else len(test_document)
        arg_sorted_scores = np.argsort(lex_scores)[:num_out_sents]
        # get the corrresponding sentence
        predicted_output = []
        for index in arg_sorted_scores:
            top_sentence = test_document[index]
            predicted_output.append(top_sentence)
        # store the indices and predicted sentences
        predicted_labels.append(arg_sorted_scores)
        predicted_outputs.append(predicted_output)
    # run rouge evaluation
    run_rouge_evaluation(test_true_summary, predicted_outputs, test_true_labels, predicted_labels)

if __name__ == "__main__":
    print("running lexrank baseline")
    run_lexrank_baseline()
