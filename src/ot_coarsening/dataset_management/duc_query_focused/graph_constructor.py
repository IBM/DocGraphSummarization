#!/usr/bin/env python3
import torch
import numpy as np
import time
import os
import nltk
import sys
sys.path.append(os.environ["GRAPH_SUM"])
from src.ot_coarsening.dataset_managment.transformer_embedding import *
from transformers import BertTokenizerFast, BertModel, DistilBertTokenizerFast, DistilBertModel, LongformerModel, LongformerTokenizerFast
from sentence_transformers import SentenceTransformer
from torch_geometric.data import Data
from transformers import logging as hf_logging
nltk.download('stopwords')
from nltk.corpus import stopwords
hf_logging.set_verbosity_error()

device = torch.device('cpu') #torch.device("cuda" if torch.cuda.is_available() else 'cpu')
stop_words = stopwords.words('english') 

class DUCGraphConstructor(GraphConstructor):

    def __init__(self):
        self.bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)
        self.bert_tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        self.sentence_transformer = SentenceTransformer('paraphrase-distilroberta-base-v1')
        self.document_transformer = LongformerModel.from_pretrained('allenai/longformer-base-4096')
        self.document_tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096')
        self.word_embedding_size = self.bert_model.config.hidden_size
        self.longform_embedding = False
    
    def _get_document_unique_words(self, document_tfidf):
        return list(document_tfidf.keys()) 

    def _get_cluster_unique_words(self, cluster_tfidf):
        # go through all the words in the document_tfidf
        unique_words = []
        for document_name in cluster_tfidf["document_tfidf"]:
            document_tfidf = cluster_tfidf["document_tfidf"][document_name]
            document_unique_words = self._get_document_unique_words(document_tfidf)
            unique_words.extend(document_unique_words)

        return unique_words

    def _filter_stop_words(self, unique_words):
        # go through each unique word and filter the stop words
        filtered_words = [word for word in unique_words if not word in stop_words]
        return filtered_words

    def _get_mean_word_embeddings(self, text, unique_words):
        # combine the lists from text into one large list
        text_list = []
        for document in text:
            text_list.extend(document)
        text = text_list
	    # perform word_embeddinng on the batch
        words_to_embeddings = compute_bert_word_embedding(self.bert_model, 
                                                      self.bert_tokenizer, 
                                                      text,
                                                      unique_words,
                                                      self.word_embedding_size)
        # mean each list
        mean_embeddings = {word: None for word in unique_words}
        for word in unique_words:
            if word in words_to_embeddings and len(words_to_embeddings[word]) > 0:
                embeddings = torch.stack(words_to_embeddings[word])
                mean_word_embedding = torch.mean(embeddings, dim=0)
                mean_embeddings[word] = mean_word_embedding
            else:
                mean_embeddings[word] = torch.zeros(self.word_embedding_size).to(device)

        return mean_embeddings

    def _get_document_embeddings(self, documents, document_sentence_embeddings, mean_sentences=True):
        if mean_sentences:
            output_embeddings = {}
            for document in document_sentence_embeddings:
                sentence_embeddings = document_sentence_embeddings[document]
                mean_embeddings = torch.mean(sentence_embeddings, dim=0)
                output_embeddings[document] = mean_embeddings
            return output_embeddings
            
        # generate zeros for each document currently
        document_embeddings = {} 
        for document_name in documents:
            document = documents[document_name]
            embedding = compute_bert_document_embedding(document, self.document_transformer, self.document_tokenizer)
            document_embeddings[document_name] = embedding
        return document_embeddings

    def _make_edges(self, word_to_node_index, sentence_to_node_index, document_to_node_index, tfidf, filtered_words):
        # edge_attr is the tfidf features
        edge_attr = []
        # edge_index is of shape (2, num_edges) in COO format
        edge_index = []
        # Undirected edges have two edges for each direction
        for document_name in document_to_node_index.keys():
            document_tfidf = tfidf["document_tfidf"][document_name]  
            sentence_tfidf = tfidf["sentence_tfidf"][document_name]  
            document_node_index = document_to_node_index[document_name]
            for sentence_index in sentence_tfidf.keys():
                sentence_node_index = sentence_to_node_index[sentence_index]
                # get sentence tfidf ict
                sentence_tfidf_dict = sentence_tfidf[sentence_index]
                sentence_words = sentence_tfidf_dict.keys()
                for word in sentence_words:
                    if not word in filtered_words:
                        continue
                    # make an edge connecting the word to the current sentence
                    word_node_index = word_to_node_index[word]
                    # make two sided edge
                    direction_one = [word_node_index, sentence_node_index]
                    direction_two = [sentence_node_index, word_node_index]
                    edge_index.append(direction_one)
                    edge_index.append(direction_two)
                    # add the corresponding attribute
                    attribute = sentence_tfidf_dict[word]
                    edge_attr.append(attribute)
                    edge_attr.append(attribute)
                    # make an edge connnecting the word to the document
                    direction_one = [word_node_index, document_node_index] 
                    direction_two = [document_node_index, word_node_index] 
                    edge_index.append(direction_one)
                    edge_index.append(direction_two)
                    # add the attribute from the
                    attribute = document_tfidf[word]
                    edge_attr.append(attribute)
                    edge_attr.append(attribute)
        # convert to numpy
        edge_index = torch.LongTensor(edge_index).T
        edge_attr = torch.FloatTensor(edge_attr).unsqueeze(-1)
        return edge_index, edge_attr

    """
        Make node attributes from word embeddings and sentence embeddings
    """
    def _make_nodes(self, word_embeddings, sentence_embeddings, document_embeddings):
        # unpack words
        word_embedding_values = torch.stack(list(word_embeddings.values()))
        word_embedding_values = word_embedding_values.to(device)
        word_embedding_keys = list(word_embeddings.keys())
        num_words = len(word_embedding_keys)
        # unpack sentences
        num_sentences = sum([len(sentence_embeddings[document_name]) for document_name in sentence_embeddings.keys()])
        sentence_embedding_values = torch.cat(list(sentence_embeddings.values()))
        # unpack documents
        document_embedding_values = torch.stack(list(document_embeddings.values()))
        document_embedding_keys = list(document_embeddings.keys())
        # make sure word and sentence embeddings have the same shape
        # assert np.shape(word_embedding_values)[-1] == np.shape(sentence_embeddings)[-1] == np.shape(document_embedding_values)[-1]
        # sentences are first then words
        attribute_matrix = torch.cat((sentence_embedding_values, word_embedding_values, document_embedding_values), dim=0)
        # output is of shape (num_nodes, num_node_features)
        # make sentence to index map
        sentence_to_node_index = {str(i): i for i in range(num_sentences)}
        # make word to node index map
        word_to_node_index = {word_embedding_keys[i]: i + num_sentences for i in range(num_words)}
        # make document to node index map
        num_documents = len(document_embedding_keys)
        document_to_node_index = {document_embedding_keys[i]: i + num_sentences + num_words for i in range(num_documents)}

        return attribute_matrix, word_to_node_index, sentence_to_node_index, document_to_node_index

    def construct_graph(self, tfidf, label):
        # label has the shape
        # {
        #    "name": "cluster name",
        #    "text": {"doc_name": ["sentence 1", "sentence 2"]},
        #    "summary": {"summerizer id": ["summary sentence 1", "summary sentence 2"]}
        #    "topic": {"title": "Title", "narr": "Query", "granularity": "granularity level"
        # }
        # tfidf has the shape
        # {
        #   "name": "cluster name",
        #   "document_tfidf": {
        #        "document_name": {
        #           "word": tfidf
        #        }
        #   }
        #   "sentence_tfidf": {
        #       "document_name": {
        #           "0": {"word": tfidf}
        #       }
        #   } 
        # }
        # Get each unique word in the document cluster
        cluster_unique_words = self._get_cluster_unique_words(tfidf)
        # Filter stop words
        filtered_unique_words = self._filter_stop_words(cluster_unique_words)
        documents = label["text"]
        # Get a word embededing for each instance of a word (dictionary word:embedding)
        word_embeddings = self._get_mean_word_embeddings(list(documents.values()), filtered_unique_words)
        # Make a dictionary of sentence embeddings document_name : (num_sentences, embedding_size)
        document_sentence_embeddings = {}
        for document_name in documents:
            document = documents[document_name]
            sentence_embeddings = compute_bert_sentence_embedding(document, self.sentence_transformer)
            document_sentence_embeddings[document_name] = sentence_embeddings
        # Make document embeddings
        document_embeddings = self._get_document_embeddings(documents, document_sentence_embeddings)
        # Make node attributes
        node_attributes, word_to_node_index, sentence_to_node_index, document_to_node_index = self._make_nodes(word_embeddings, document_sentence_embeddings, document_embeddings)
        # Make edges
        edge_index, edge_attributes = self._make_edges(word_to_node_index,
                                                       sentence_to_node_index,
                                                       document_to_node_index,
                                                       tfidf,
                                                       filtered_unique_words)
        # Make data and return
        data_object = Data(x=node_attributes,
                           edge_index=edge_index,
                           edge_attr=edge_attributes,
                           label=label,
                           tfidf=tfidf,
                           num_sentences=len(label["text"]))

        return data_object


