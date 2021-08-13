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
def compute_bert_sentence_embedding(sentences, model, prepend_ordering=False):
    model = model.to(device)
    sentence_embeddings = model.encode(sentences)
    sentence_embeddings = torch.Tensor(sentence_embeddings).to(device)
    if prepend_ordering:
        # replace the first float wiht a proprotion of how far through the sentence is in the paragraph
        indices = torch.arange(sentence_embeddings.shape[0]).float()
        indices /= sentence_embeddings.shape[0]
        for i in range(20):
            sentence_embeddings[:, i] = indices
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
    document_embedding = torch.mean(document_embedding, dim=0)
    return document_embedding    


