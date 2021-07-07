import os
import argparse
import json
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import normalize
from io import StringIO
from lxml import etree
import jsonlines
from bs4 import BeautifulSoup

root_path = os.path.join(os.environ["DATA_PATH"], "DUC")

"""
    TFIDF processing code from heter_sum_grapdf_embedding(text):

    :param text: list, doc_number * word
    :return: 
        vectorizer: 
            vocabulary_: word2id
            get_feature_names(): id2word
        tfidf: array [sent_number, max_word_number]

    vectorizer = CountVectorizer(lowercase=True)
    word_count = vectorizer.fit_transform(text)
    tfidf_transformer = TfidfTransformer()
    tfidf = tfidf_transformer.fit_transform(word_count)
    tfidf_weight = tfidf.toarray()
    return vectorizer, tfidf_weighth 
"""
"""
    Computes tfidf features for the DUC2005-2007 datasets
    file structure:
    uncompressed/
        DUC2006
            cluster
                articles
                    xml
        DUC2006
        DUC2007
"""
"""
JSON structure
for each of DUC2005-2007
{
   "cluster_name":{"document_name":{"sentence_index":tfidf_value}}}
}
"""

def save_json_lines(file_name, json_lines_list):
    with jsonlines.open(file_name, mode='w') as writer:
        writer.write_all(json_lines_list) 

def xml_to_sentences(document_path):
    copy = False
    text = False
    text_lines = []
    with open(document_path, encoding="ISO-8859-1") as infile: 
        for line in infile:
            if line.strip() == "<TEXT>":
                copy = True
                text = True
                continue
            elif line.strip() == "</TEXT>":
                copy = False
                text = False
                continue
            if line.strip() == "<P>" and not text:
                copy = True
                continue
            elif line.strip() == "</P>" and not text:
                copy = False
                continue
            if copy:
                text_lines.append(line)
    # map to sentences 
    file_string = "".join(text_lines)
    file_string = file_string.replace("<P>", "")
    file_string = file_string.replace("</P>", "")
    file_string = file_string.replace("\n", " ")
    file_string = file_string.lower()
    sentences = file_string.split(".")
    sentences = [sentence for sentence in sentences if len(sentence.strip()) != 0]

    return sentences

def compress_array(a, id2word):
    """
    :param a: matrix, [N, M], N is document number, M is word number
    :param id2word: word id to word
    :return: 
    """
    d = {}
    for i in range(len(a)):
        d[i] = {}
        for j in range(len(a[i])):
            if a[i][j] != 0 and j in id2word:
                d[i][id2word[j]] = a[i][j]
    return d

def compute_sentence_tfidf(document_name, cluster_label):
    sentence_tfidf = {} 
    # get sentences
    text = cluster_label["text"][document_name]
    # compute tfidf
    vectorizer = CountVectorizer(lowercase=True)
    word_count = vectorizer.fit_transform(text)
    tfidf_transformer = TfidfTransformer()
    tfidf = tfidf_transformer.fit_transform(word_count)
    tfidf_weight = tfidf.toarray()
    id2word = {} 
    for w, tfidf_id in vectorizer.vocabulary_.items():   # word -> tfidf matrix row number
        id2word[tfidf_id] = w
    tfidfvector = compress_array(tfidf_weight, id2word)
    return tfidfvector

def compute_document_tfidf(dataset_name, document_name, cluster_label):
    dataset_root_path = os.path.join(root_path, "unprocessed", dataset_name)
    document_tfidf = {} 
    # get the sentences from all documents and aggregate them 
    texts = cluster_label["text"]
    all_text = [inner for outer in texts.values()
                for inner in outer]
    # compute the tfidf for the specific document
    current_document = cluster_label["text"][document_name]
    current_vectorizer = CountVectorizer(lowercase=True)
    word_count = current_vectorizer.fit_transform(current_document)
    current_vocab = current_vectorizer.vocabulary_.items()
    current_vocab, _ = zip(*current_vocab)
    # compute tfidf
    vectorizer = CountVectorizer(lowercase=True)
    word_count = vectorizer.fit_transform(all_text)
    tfidf_transformer = TfidfTransformer()
    tfidf = tfidf_transformer.fit_transform(word_count)
    tfidf_weight = tfidf.toarray()
    id2word = {} 
    for w, tfidf_id in vectorizer.vocabulary_.items():   # word -> tfidf matrix row number
        if w in current_vocab:
            id2word[tfidf_id] = w
    tfidfvector = compress_array(tfidf_weight, id2word)
    return tfidfvector

"""
    Given a cluster name it returns the human created
    summaries associated with that cluster.
"""
def get_cluster_summaries(cluster_name, summary_root_path):
    # cluster prefix d301i : prefix = D301
    cluster_prefix = cluster_name[0:4].upper()
    # get the cluster specific summaries
    summary_path_names = os.listdir(summary_root_path)
    cluster_summary_paths = [path for path in summary_path_names if path[0:4] == cluster_prefix]
    # get the summarizer name A-J
    summarizer_names = [path[13] for path in cluster_summary_paths]
    # make the output dictionary
    cluster_summaries = {}
    for i in range(len(summarizer_names)):
        summarizer_file_name = cluster_summary_paths[i]
        summarizer_name = summarizer_names[i]
        # open the file and save the contents
        summarizer_path = os.path.join(summary_root_path, summarizer_file_name)
        with open(summarizer_path, "r", encoding="ISO-8859-1") as summary_file:
            sentences = []
            for line in summary_file:
                line = line.strip()
                sentences.append(line)
        cluster_summaries[summarizer_name] = sentences
    
    return cluster_summaries

def get_cluster_topic(dataset_root_path, dataset_name, cluster_name):
    # load the xml file
    topic_path = os.path.join(dataset_root_path, dataset_name.lower() + "_topics.sgml") 
    with open(topic_path, "r") as xml_doc:
        # parse the xml 
        soup = BeautifulSoup(xml_doc, 'lxml')
        # load each topic object
        topics = soup.find_all("topic")
        cluster_topic = None
        for topic in topics:
            num = topic.find("num").get_text().lower().strip()
            if num == cluster_name.lower():
                cluster_topic = topic
                break

        if cluster_topic is None:
            raise Exception("cluster topic is none, should exist")
        # parse the topic to json
        topic_json = {}
        topic_json["title"] = topic.find("title").get_text().strip()
        topic_json["narr"] =  topic.find("narr").get_text().strip().replace("\n", " ").lower()
        if len(topic.find("granularity")) > 0:
            topic_json["granularity"] = topic.find("granularity").get_text().strip()


    return topic_json

"""
    Makes a label associated with a cluster
    
    { 
      "text": {"document_name": sentences}
      "summaries": {"summarizer A": summary}
    }
"""
def make_cluster_label(cluster_name, dataset_root_path, dataset_name):
    # get cluster vocabulary counts
    cluster_label_dict = {"name": cluster_name}
    # make a text dict for each document
    cluster_label_dict["text"] = {}
    # dataset year DUC2005, DUC2006, DUC2007
    dataset_year = dataset_name[3:]
    # get the summary root path
    summary_root_path = os.path.join(dataset_root_path, dataset_year + "Results", "ROUGE", "models") 
    # get document paths
    document_sub_path = dataset_name.lower() + "_docs"
    cluster_doc_path = os.path.join(dataset_root_path, document_sub_path, cluster_name)
    document_paths = os.listdir(cluster_doc_path)
    for document_name in document_paths:
        document_path = os.path.join(cluster_doc_path, document_name)
        # load the xml file
        sentences = xml_to_sentences(document_path)
        # add the sentences to the cluster label
        cluster_label_dict["text"][document_name] = sentences 
    # add the summaries to the cluster label
    summaries = get_cluster_summaries(cluster_name, summary_root_path)
    cluster_label_dict["summary"] = summaries 
    # add the topic
    topic = get_cluster_topic(dataset_root_path, dataset_name, cluster_name)
    cluster_label_dict["topic"] = topic

    return cluster_label_dict

def make_cluster_tfidf(cluster_name, dataset_root_path, cluster_label):
    cluster_tfidf = {"name": cluster_name}
    # compute the total cluster vocab
    # get document paths
    document_sub_path = dataset_name.lower() + "_docs"
    cluster_doc_path = os.path.join(dataset_root_path, document_sub_path, cluster_name)
    document_paths = os.listdir(cluster_doc_path)
    for document_name in document_paths:
        # compute the document tfidf
        document_tfidf = compute_document_tfidf(dataset_name, document_name, cluster_label)
        cluster_tfidf["document_tfidf"] = document_tfidf
        # compute the sentence tfidf
        sentence_tfidf = compute_sentence_tfidf(document_name, cluster_label)
        cluster_tfidf["sentence_tfidf"] = sentence_tfidf
    
    return cluster_tfidf

def process_data(dataset_name):
    # process the given dataset name
    tfidf_lines = []
    label_lines = []
    # get the dataset path
    dataset_root_path = os.path.join(root_path, "uncompressed", dataset_name)
    document_sub_path = dataset_name.lower() + "_docs"
    dataset_clusters = os.path.join(dataset_root_path, document_sub_path)
    # get the cluster directory names
    clusters = os.listdir(dataset_clusters) 
    for cluster_name in clusters:
        # make cluster labels
        cluster_label = make_cluster_label(cluster_name, dataset_root_path, dataset_name) 
        # make cluster tfidf
        cluster_tfidf = make_cluster_tfidf(cluster_name, dataset_root_path, cluster_label)
        # add to lists
        label_lines.append(cluster_label)
        tfidf_lines.append(cluster_tfidf)
    # save the files
    unprocessed_save_dir = os.path.join(root_path, "unprocessed", dataset_name)
    save_json_lines(os.path.join(unprocessed_save_dir, "tfidf.jsonl"), tfidf_lines)
    save_json_lines(os.path.join(unprocessed_save_dir, "label.jsonl"), label_lines)

if __name__ == "__main__":
    datasets = ["DUC2005"]#, "DUC2006", "DUC2007"]
    for dataset_name in datasets:
        process_data(dataset_name)
   
