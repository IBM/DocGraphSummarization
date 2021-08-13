#!/usr/bin/env python3
import torch
import numpy as np
import os
import sys
sys.path.append(os.environ["GRAPH_SUM"])
from src.ot_coarsening.dataset_management.transformer_embedding import *
import time
from transformers import BertTokenizerFast, BertModel, DistilBertTokenizerFast, DistilBertModel, LongformerModel, LongformerTokenizerFast
from sentence_transformers import SentenceTransformer
from torch_geometric.data import Data
from transformers import logging as hf_logging
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
hf_logging.set_verbosity_error()

device = torch.device('cpu') #torch.device("cuda" if torch.cuda.is_available() else 'cpu')
stop_words = stopwords.words('english') 

"""
    This is an object that gets passed to a graph dataset that gets used to
    construct a graph
"""
class CNNDailyMailGraphConstructor():

    def __init__(self, similarity=False, prepend_ordering=False):
        self.bert_model = LongformerModel.from_pretrained('allenai/longformer-base-4096')
        self.bert_tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096')
        self.sentence_transformer = SentenceTransformer('paraphrase-distilroberta-base-v1')
        self.word_embedding_size = self.bert_model.config.hidden_size
        self.similarity = similarity
        self.prepend_ordering = prepend_ordering
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

    def _get_unique_words(self, tfidf):
        unique_words = set()
        for sentence_id in tfidf.keys():
            sentence = tfidf[sentence_id]
            words = list(sentence.keys())
            unique_words.update(words)

        return list(unique_words)

    def _filter_stop_words(self, unique_words):
        # go through each unique word and filter the stop words
        filtered_words = [word for word in unique_words if not word in stop_words]
        return filtered_words

    def _get_mean_word_embeddings(self, text, unique_words):
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

    def _make_edges(self, word_to_node_index, sentence_to_node_index,  tfidf, filtered_words):
        # edge_attr is the tfidf features
        edge_attr = []
        # edge_index is of shape (2, num_edges) in COO format
        edge_index = []
        # Undirected edges have two edges for each direction
        for sentence_index in tfidf.keys():
            sentence_node_index = sentence_to_node_index[sentence_index]
            # get sentence tfidf ict
            tfidf_dict = tfidf[sentence_index]
            sentence_words = tfidf_dict.keys()
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
                attribute = tfidf_dict[word]
                edge_attr.append(attribute)
                edge_attr.append(attribute)
        # convert to numpy
        edge_index = torch.LongTensor(edge_index).T
        edge_attr = torch.FloatTensor(edge_attr).unsqueeze(-1)
        return edge_index, edge_attr

    def _make_similarity_graph_edges(self, sentence_to_node_index, sentence_embeddings):
        # edge_attr is the tfidf features
        edge_attr = []
        # edge_index is of shape (2, num_edges) in COO format
        edge_index = []
        # Undirected edges have two edges for each direction
        for sentence_index in sentence_to_node_index.keys():
            sentence_node_index = sentence_to_node_index[sentence_index]
            for other_sentence_index in sentence_to_node_index.keys():
                if other_sentence_index == sentence_index:
                    continue
                other_sentence_node_index = sentence_to_node_index[other_sentence_index]
                # make an edge between the two sentences
                edge = [sentence_node_index, other_sentence_node_index]
                # other_edge = [other_sentence_node_index, sentence_node_index]
                edge_index.append(edge)
                # edge_index.append(other_edge)
                # calculate the cosine similarity
                sentence_one_embedding = sentence_embeddings[sentence_node_index]
                sentence_two_embedding = sentence_embeddings[other_sentence_node_index]
                cosine_similarity = self.cosine_similarity(sentence_one_embedding, sentence_two_embedding)
                cosine_similarity = (cosine_similarity + 1) / 2
                edge_attr.append(cosine_similarity)
                # edge_attr.append(cosine_similarity)
        # convert to numpy
        edge_index = torch.LongTensor(edge_index).T
        edge_attr = torch.FloatTensor(edge_attr).unsqueeze(-1)
        return edge_index, edge_attr

    """
        Make node attributes from word embeddings and sentence embeddings
    """
    def _make_nodes(self, word_embeddings, sentence_embeddings):
        num_sentences = np.shape(sentence_embeddings)[0]
        # sentences are first then words
        if not self.similarity:
            # make sure word and sentence embeddings have the same shape
            #assert np.shape(word_embedding_values)[-1] == np.shape(sentence_embeddings)[-1]
            word_embedding_values = torch.stack(list(word_embeddings.values()))
            word_embedding_values = word_embedding_values.to(device)
            word_embedding_keys = list(word_embeddings.keys())
            num_words = len(word_embedding_keys)
            attribute_matrix = torch.cat((sentence_embeddings, word_embedding_values), dim=0)
            # output is of shape (num_nodes, num_node_features)
            # make sentence to index map
            sentence_to_node_index = {str(i): i for i in range(num_sentences)}
            # make word to node index map
            word_to_node_index = {word_embedding_keys[i]: i + num_sentences for i in range(num_words)}
        else:
            # similarity graph
            attribute_matrix = sentence_embeddings
            # make sentence to index map
            sentence_to_node_index = {str(i): i for i in range(num_sentences)}
            word_to_node_index = None

        return attribute_matrix, word_to_node_index, sentence_to_node_index

    def _check_word_exists(self, tfidf, label):
        assert len(tfidf.keys()) == len(label["text"])
        for sentence_index in range(len(label["text"])):
            sentence = label["text"][sentence_index]
            tfidf_sentence = tfidf[str(sentence_index)]
            for word in tfidf_sentence:
                word_exists = False
                if word in sentence:
                    word_exists = True
                    break
                if not word_exists:
                    print(word)
                assert word_exists

    def construct_graph(self, tfidf, label):
        # sanity check
        #self._check_word_exists(tfidf, label)
        # label has the shape
        # {
        #    "text": [],
        #    "summary": [],
        #    "label": []
        # }
        # tfidf has the shape
        # {
        #    "0": {
        #        "word": tfidf
        #        }
        # }
        # Get each unique word in the document
        unique_words = self._get_unique_words(tfidf)
        # Filter stop words
        filtered_unique_words = self._filter_stop_words(unique_words)
        # Get a word embededing for each instance of a word (dictionary word:embedding)
        if not self.similarity:
            word_embeddings = self._get_mean_word_embeddings(label["text"], filtered_unique_words)
        else:
            word_embeddings = None
        # Make a list of sentence embeddings (num_sentences, embedding_size)
        sentence_embeddings = compute_bert_sentence_embedding(label["text"], self.sentence_transformer, prepend_ordering=self.prepend_ordering)
        # Make node attributes
        node_attributes, word_to_node_index, sentence_to_node_index = self._make_nodes(word_embeddings, sentence_embeddings)
        # Make edges
        if not self.similarity:
            edge_index, edge_attributes = self._make_edges(word_to_node_index,
                                                           sentence_to_node_index,
                                                           tfidf,
                                                           filtered_unique_words)
        else:
            # make similarity graph edges
            edge_index, edge_attributes = self._make_similarity_graph_edges(sentence_to_node_index, sentence_embeddings)
        # get labels
        if "label" in label:
            labels = torch.Tensor(label["label"])
            # filter invalid graphs
            if len(label["text"]) < len(label["label"]):
                return None
        else: 
            labels = None
        # setup the rankings
        if "average_rankings" in label:
            rankings = label["average_rankings"] 
        else:
            rankings = None
        # Make data and return
        data_object = Data(x=node_attributes,
                           edge_index=edge_index,
                           edge_attr=edge_attributes,
                           y=labels,
                           label=label,
                           rankings=rankings,
                           tfidf=tfidf,
                           num_sentences=len(label["text"]))

        return data_object

