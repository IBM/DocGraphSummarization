#!/usr/bin/env python3
import torch
import numpy as np
import os
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

def compute_bert_tokenization(tokenizer, sentence, padding=False):
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
            embedding = embeddings[j]
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

"""
    Returns two outputs a list of words and a list of embeddings 
    corresponding to each word. This includes duplicates of words
"""
def compute_bert_word_embeddings_token_pairs(bert_model, bert_tokenizer, sentences, unique_words, word_embedding_size, layers=[-1, -2, -3, -4], longformer=True):
    if longformer: 
        # merge all sentences into one
        single_sentence = " ".join(sentences)
        sentences = [single_sentence]
    all_embeddings = []
    all_tokens = []
    for sentence in sentences:
        encoded = bert_tokenizer.encode_plus(sentence, add_special_tokens=False) 
        input_ids = torch.LongTensor(encoded["input_ids"])[None, :]
        output = bert_model(input_ids=input_ids)
        states = output.last_hidden_state.squeeze()
        tokens = bert_tokenizer.convert_ids_to_tokens(encoded["input_ids"])
        states, tokens = merge_subword_embeddings(states, tokens)
        all_embeddings.append(states)
        all_tokens.append(tokens)

    return all_embeddings, all_tokens

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
        #embeddings, tokens = drop_first_last(token_embeddings, tokens)
        # merge subword embeddings
        #embeddings, tokens = merge_subword_embeddings(embeddings, tokens)
        # remove punctionation
        #embeddings, tokens = remove_punctuation(embeddings, tokens)
        # add each instance of a word's embedding to the embedding list
        for token_index in range(len(tokens)):
            token = tokens[token_index]
            if not token in unique_words:
                # this is necessary to filter stop words
                continue
            token_embedding = token_embeddings[token_index] 
            if len(token_embedding.shape) == 0:
                print(token_embedding.shape)
                continue
            words_to_embeddings[token].append(token_embedding)

    filtered_words_to_embeddings = {word: words_to_embeddings[word] for word in words_to_embeddings if len(words_to_embeddings[word]) > 0}

    return filtered_words_to_embeddings

"""
    Generates sentence embeddings for a sentence using pretrained
    BERT.
"""
def compute_bert_sentence_embedding(sentences, model):
    model = model.to(device)
    sentence_embeddings = model.encode(sentences)
    sentence_embeddings = torch.Tensor(sentence_embeddings).to(device)
    return sentence_embeddings

"""
    This function takes a model like Longformer
    and returns the document embedding for the collection of sentences.

"""
def compute_bert_document_embedding(sentences, model, tokenizer):
    # compute the bert document embedding for a single document
    # apply the longformer to the entire document
    combined_sentences = ".".join(sentences)
    token_ids = tokenizer(combined_sentences)["input_ids"]
    document_embedding = model(token_ids).last_hidden_state.squeeze()
    print(document_embedding.shape)
    document_embedding = torch.mean(document_embedding, dim=0)
    return document_embedding    


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

    def __init__(self, similarity=False):
        self.bert_model = LongformerModel.from_pretrained('allenai/longformer-base-4096')
        self.bert_tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096')
        self.sentence_transformer = SentenceTransformer('paraphrase-distilroberta-base-v1')
        self.word_embedding_size = self.bert_model.config.hidden_size
        self.similarity = similarity
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
                edge_index.append(edge)
                # calculate the cosine similarity
                sentence_one_embedding = sentence_embeddings[sentence_node_index]
                sentence_two_embedding = sentence_embeddings[other_sentence_node_index]
                cosine_similarity = self.cosine_similarity(sentence_one_embedding, sentence_two_embedding)
                edge_attr.append(cosine_similarity)
                print("cosine similarity shape")
                print(cosine_similarity.shape)
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
            assert np.shape(word_embedding_values)[-1] == np.shape(sentence_embeddings)[-1]
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
        sentence_embeddings = compute_bert_sentence_embedding(label["text"], self.sentence_transformer)
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


