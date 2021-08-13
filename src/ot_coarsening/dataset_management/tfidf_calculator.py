import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import jsonlines

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
    if not document_name is None:
        text = cluster_label["text"][document_name]
    else:
        text = cluster_label["text"]
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
    out_dict = {}
    for key in list(tfidfvector.keys()):
        out_dict[str(key)] = tfidfvector[key]
    return out_dict

def compute_document_tfidf(document_name, cluster_label):
    dataset_root_path = os.path.join(root_path, "unprocessed", dataset_name)
    document_tfidf = {} 
    # document name index
    texts = cluster_label["text"]
    document_names = list(texts.keys())
    document_name_index = 0
    for index, name in enumerate(document_names):
        if name == document_name:
            document_name_index = index
            break
    # get the sentences from all documents and aggregate them 
    all_text = []
    for doc_name in texts.keys():
        doc_text = texts[doc_name]
        # merge all sentences in the doc
        merged = ".".join(doc_text)
        all_text.append(merged)
    vectorizer = CountVectorizer(lowercase=True)
    word_count = vectorizer.fit_transform(all_text)

