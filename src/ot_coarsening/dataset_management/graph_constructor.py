#!/usr/bin/env python3
import torch
import numpy as np
import os
import time
from transformers import BertTokenizerFast, BertModel, DistilBertTokenizerFast, DistilBertModel, LongformerModel 
from sentence_transformers import SentenceTransformer
from torch_geometric.data import Data
from transformers import logging as hf_logging
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
hf_logging.set_verbosity_error()

device = torch.device('cpu') #torch.device("cuda" if torch.cuda.is_available() else 'cpu')
stop_words = stopwords.words('english') 

def compute_bert_tokenization(tokenizer, sentence, padding=True):
    assert isinstance(sentence, str)
    tokens = tokenizer.tokenize(sentence, padding=padding)
    input_ids = torch.tensor(tokenizer.encode(sentence, padding)).unsqueeze(0)
    return tokens, input_ids

# drop beggining and end embeddings
def drop_first_last(embeddings, tokens):
    return embeddings[1:-1, :], tokens[1:-1]

def merge_subword_embeddings(embeddings, tokens):
    output_embeddings = []
    output_tokens = []
    i = 0
    while i < len(tokens):
        current_embedding = embeddings[i]
        current_token = tokens[i]
        following_embeddings = [current_embedding]
        following_tokens = [current_token]
        # while there are subwords following it
        j = i + 1
        while j < len(tokens):
            token = tokens[j]
            embedding = embeddings[i]
            if token[0:2] == "##":
                following_embeddings.append(embedding)
                following_tokens.append(token[2:])
            else:
                break
            j += 1
        # skip number based on how many are subwords
        i = j
        # merge following embeddings
        following_embeddings = torch.stack(following_embeddings)
        merged_embedding = torch.mean(following_embeddings, dim=0)
        # merge following tokens
        merged_token = ''.join(following_tokens)
        # add to output
        output_tokens.append(merged_token)
        output_embeddings.append(merged_embedding)

    output_embeddings = torch.stack(output_embeddings)
    return output_embeddings, output_tokens

def remove_punctuation(embeddings, tokens):
    output_embeddings = []
    output_tokens = []

    for i in range(len(embeddings)):
        token = tokens[i]
        embedding = embeddings[i]
        if token.isalnum():
            output_embeddings.append(embedding)
            output_tokens.append(token)

    output_embeddings = torch.stack(output_embeddings)
    return output_embeddings, output_tokens

def convert_to_average_duplicates(embeddings, tokens):
    # make a dictionary of unique words in tokens to lists
    average_dictionary = {}
    for token in tokens:
        average_dictionary[token] = []
    # go through and add each unique embedding for a word to its corresponding dictionary list
    for i in range(len(embeddings)):
        token = tokens[i]
        embedding = embeddings[i]
        average_dictionary[token].append(embedding)
    # average each list
    for word, embedding_list in average_dictionary.items():
        embedding_list = torch.stack(embedding_list)
        mean_embedding_value = torch.mean(embedding_list, dim=0)
        average_dictionary[word] = mean_embedding_value
    # return
    return average_dictionary

"""
    Returns two outputs a list of words and a list of embeddings 
    corresponding to each word. This includes duplicates of words
"""
def compute_bert_word_embeddings_token_pairs(bert_model, bert_tokenizer, sentences, unique_words, word_embedding_size):
    # perform batch tokenizations
    batch_encoding = bert_tokenizer(sentences, padding=True)
    # deduce the tokens from the ids in batch_encoding
    tokens_list = []
    for i in range(len(batch_encoding.input_ids)):
        token_ids = batch_encoding.input_ids[i]
        tokens =  bert_tokenizer.convert_ids_to_tokens(token_ids)
        tokens_list.append(tokens)
    # go through and produce the word embeddings
    input_ids = torch.LongTensor(batch_encoding.input_ids).to(device)
    attention_masks = torch.FloatTensor(batch_encoding.attention_mask).to(device)
    #token_type_ids = torch.LongTensor(batch_encoding.token_type_ids).to(device)
    base_model_output = bert_model(input_ids=input_ids, attention_mask=attention_masks)#, token_type_ids=token_type_ids)
    embeddings = base_model_output.last_hidden_state
    # apply attention mask to embeddings
    output_embeddings = []
    output_tokens = []
    for i, embedding_list in enumerate(embeddings):
        attention_mask = attention_masks[i] > 0
        masked_embeddings = torch.masked_select(embedding_list.T, attention_mask).T
        masked_embeddings = torch.reshape(masked_embeddings, (-1, word_embedding_size))
        output_embeddings.append(masked_embeddings)
        # do token mask
        tokens = []
        for j, token in enumerate(tokens_list[i]):
            if attention_mask[j]:
                tokens.append(token)
        output_tokens.append(tokens)
    
    return output_embeddings, output_tokens

"""
    Generates word embeddings for words in a given sentenece using
    pretrained BERT.

    - sentence
"""
def compute_bert_word_embedding(bert_model, bert_tokenizer, sentences, unique_words, word_embedding_size):
    words_to_embeddings = {word: [] for word in unique_words}
    # tokenize the sentence (includes punctuation, before and after tokens, and subword tokens)
    output_embeddings, output_tokens = compute_bert_word_embeddings_token_pairs(bert_model, bert_tokenizer, sentences, unique_words, word_embedding_size)
    # clean up the output
    for i in range(len(output_embeddings)):
        tokens = output_tokens[i]
        token_embeddings = output_embeddings[i] 
        # drop first and last
        embeddings, tokens = drop_first_last(token_embeddings, tokens)
        # merge subword embeddings
        embeddings, tokens = merge_subword_embeddings(embeddings, tokens)
        # remove punctionation
        embeddings, tokens = remove_punctuation(embeddings, tokens)
        # add each instance of a word's embedding to the embedding list
        for token_index in range(len(tokens)):
            token = tokens[token_index]
            if not token in unique_words:
                # this is necessary to filter stop words
                continue
            token_embedding = token_embeddings[token_index] 
            words_to_embeddings[token].append(token_embedding)

    return words_to_embeddings

"""
    Generates sentence embeddings for a sentence using pretrained
    BERT.
"""
def compute_bert_sentence_embedding(sentences, model):
    model = model.to(device)
    sentence_embeddings = model.encode(sentences)
    sentence_embeddings = torch.Tensor(sentence_embeddings).to(device)
    return sentence_embeddings


class GraphConstructor():
      
    def __init__(self):
        pass

    def construct_graph(self):
        pass

"""
    This is an object that gets passed to a graph dataset that gets used to
    construct a graph
"""
class CNNDailyMailGraphConstructor():

    def __init__(self):
        self.bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)
        self.bert_tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        # self.sentence_transformer =  SentenceTransformer('paraphrase-mpnet-base-v2')
        self.sentence_transformer = SentenceTransformer('paraphrase-distilroberta-base-v1')
        self.word_embedding_size = self.bert_model.config.hidden_size
        self.longform_embedding = False

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

    def _get_mean_word_embeddings(self, text, tfidf, unique_words):
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

    """
        Make node attributes from word embeddings and sentence embeddings
    """
    def _make_nodes(self, word_embeddings, sentence_embeddings):
        word_embedding_values = torch.stack(list(word_embeddings.values()))
        word_embedding_values = word_embedding_values.to(device)
        word_embedding_keys = list(word_embeddings.keys())
        num_words = len(word_embedding_keys)
        num_sentences = np.shape(sentence_embeddings)[0]
        # make sure word and sentence embeddings have the same shape
        assert np.shape(word_embedding_values)[-1] == np.shape(sentence_embeddings)[-1]
        # sentences are first then words
        attribute_matrix = torch.cat((sentence_embeddings, word_embedding_values), dim=0)
        # output is of shape (num_nodes, num_node_features)
        # make sentence to index map
        sentence_to_node_index = {str(i): i for i in range(num_sentences)}
        # make word to node index map
        word_to_node_index = {word_embedding_keys[i]: i + num_sentences for i in range(num_words)}

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
        self._check_word_exists(tfidf, label)
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
        word_embeddings = self._get_mean_word_embeddings(label["text"], tfidf, filtered_unique_words)
        # Make a list of sentence embeddings (num_sentences, embedding_size)
        sentence_embeddings = compute_bert_sentence_embedding(label["text"], self.sentence_transformer)
        # Make node attributes
        node_attributes, word_to_node_index, sentence_to_node_index = self._make_nodes(word_embeddings,
                                                                                       sentence_embeddings)
        # Make edges
        edge_index, edge_attributes = self._make_edges(word_to_node_index,
                                                       sentence_to_node_index,
                                                       tfidf,
                                                       filtered_unique_words)
        # get labels
        labels = torch.Tensor(label["label"])
        # filter invalid graphs
        if len(label["text"]) < len(label["label"]):
            return None
        # Make data and return
        data_object = Data(x=node_attributes,
                           edge_index=edge_index,
                           edge_attr=edge_attributes,
                           y=labels,
                           label=label,
                           tfidf=tfidf,
                           num_sentences=len(label["text"]))

        return data_object

class DUCGraphConstructor(GraphConstructor):

    def __init__(self):
        self.bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)
        self.bert_tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        self.sentence_transformer = SentenceTransformer('paraphrase-distilroberta-base-v1')
        self.word_embedding_size = self.bert_model.config.hidden_size
        self.longform_embedding = False

    def _get_unique_words(self, tfidf):
        # go through all the words in the document_tfidf
        document_tfidf = tfidf["document_tfidf"]
        unique_words  = list(document_tfidf.keys())

        return unique_words

    def _filter_stop_words(self, unique_words):
        # go through each unique word and filter the stop words
        filtered_words = [word for word in unique_words if not word in stop_words]
        return filtered_words

    def _get_mean_word_embeddings(self, text, tfidf, unique_words):
        words_to_embeddings = {word: [] for word in unique_words}
	    # perform word_embeddinng on the batch
        # start = time.time()
        embedding_dicts = compute_bert_word_embedding(self.bert_model, self.bert_tokenizer, text, unique_words, self.word_embedding_size)
        # finish = time.time()
        # print("Compute bert word embedding time {}".format(finish - start))
        for i, word_embeddings_dict in enumerate(embedding_dicts):
            tfidf_words = list(tfidf[str(i)].keys())
            # assuming we have a list of tokens and a list of word embeddings
            for j, token in enumerate(tfidf_words):
                if token in word_embeddings_dict and token in words_to_embeddings:
                    word_embedding = word_embeddings_dict[token]
                    words_to_embeddings[token].append(word_embedding)
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

    """
        Make node attributes from word embeddings and sentence embeddings
    """
    def _make_nodes(self, word_embeddings, sentence_embeddings):
        word_embedding_values = torch.stack(list(word_embeddings.values()))
        word_embedding_values = word_embedding_values.to(device)
        word_embedding_keys = list(word_embeddings.keys())
        num_words = len(word_embedding_keys)
        num_sentences = np.shape(sentence_embeddings)[0]
        # make sure word and sentence embeddings have the same shape
        assert np.shape(word_embedding_values)[-1] == np.shape(sentence_embeddings)[-1]
        # sentences are first then words
        attribute_matrix = torch.cat((sentence_embeddings, word_embedding_values), dim=0)
        # output is of shape (num_nodes, num_node_features)
        # make sentence to index map
        sentence_to_node_index = {str(i): i for i in range(num_sentences)}
        # make word to node index map
        word_to_node_index = {word_embedding_keys[i]: i + num_sentences for i in range(num_words)}

        return attribute_matrix, word_to_node_index, sentence_to_node_index

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
        #        "word": tfidf
        #   }
        #   "sentence_tfidf": {
        #       "0": {"word": tfidf}
        #   } 
        # }
        # Get each unique word in the document cluster
        unique_words = self._get_unique_words(tfidf)
        # Filter stop words
        filtered_unique_words = self._filter_stop_words(unique_words)
        # Get a word embededing for each instance of a word (dictionary word:embedding)
        word_embeddings = self._get_mean_word_embeddings(label["text"], tfidf, filtered_unique_words)
        # Make a list of sentence embeddings (num_sentences, embedding_size)
        sentence_embeddings = compute_bert_sentence_embedding(label["text"], self.sentence_transformer)
        # Make document embeddings
        document_embeddings = self._get_document_embeddings()
        # Make node attributes
        node_attributes, word_to_node_index, sentence_to_node_index, document_to_node_index = self._make_nodes(word_embeddings,
                                                                                                            sentence_embeddings)
        # Make edges
        edge_index, edge_attributes = self._make_edges(word_to_node_index,
                                                       sentence_to_node_index,
                                                       tfidf,
                                                       filtered_unique_words)
        # get labels
        labels = torch.Tensor(label["label"])
        # filter invalid graphs
        if len(label["text"]) < len(label["label"]):
            return None
        # Make data and return
        data_object = Data(x=node_attributes,
                           edge_index=edge_index,
                           edge_attr=edge_attributes,
                           y=labels,
                           label=label,
                           tfidf=tfidf,
                           num_sentences=len(label["text"]))

        return data_object


