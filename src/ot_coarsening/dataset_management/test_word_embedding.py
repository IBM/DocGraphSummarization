import graph_constructor
import torch
from transformers import BertTokenizerFast, BertModel, DistilBertTokenizerFast, DistilBertModel, LongformerModel 

if __name__ == "__main__":
    device = torch.device('cpu')
    bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
    bert_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    sentences = ["this is a test sentence", "this is another test sentence more words"]
    unique_words = ["this", "is", "a", "another", "test", "sentence", "more", "words"]
    word_embedding_size = 768
    embeddings, words = graph_constructor.compute_bert_word_embeddings_token_pairs(bert_model, bert_tokenizer, sentences, unique_words, word_embedding_size)

    embeddings_dicts = graph_constructor.compute_bert_word_embedding(bert_model, bert_tokenizer, sentences, unique_words, word_embedding_size)
    print(embeddings_dicts)

    print("Words")
    print(words)
    print("Embeddings")
    print(embeddings)
    
