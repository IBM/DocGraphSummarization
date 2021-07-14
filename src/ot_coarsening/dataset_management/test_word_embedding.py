import graph_constructor
import torch
from transformers import BertTokenizerFast, BertModel, DistilBertTokenizerFast, DistilBertModel, LongformerModel, LongformerTokenizerFast

if __name__ == "__main__":
    device = torch.device('cpu')
    bert_model = LongformerModel.from_pretrained('allenai/longformer-base-4096')
    bert_tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096')
 
    sentences = ["this is a test sentence .", "this is another test sentence more words ."]
    unique_words = ["this", "is", "a", "another", "test", "sentence", "more", "words"]
    word_embedding_size = 768
    embeddings, words = graph_constructor.compute_bert_word_embeddings_token_pairs(bert_model, bert_tokenizer, sentences, unique_words, word_embedding_size)

    #embeddings_dicts = graph_constructor.compute_bert_word_embedding(bert_model, bert_tokenizer, sentences, unique_words, word_embedding_size)
    #print(embeddings_dicts)

    print("Words")
    print(words)
    
