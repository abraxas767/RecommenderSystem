from collections import Counter
from copy import deepcopy
import numpy as np
from stopwords import stopwords_en
from nltk.stem import PorterStemmer

def tf(nparray) -> dict:
    # find maximum frequency
    max_freq = nparray.max()
    # get normalized vector
    return nparray / max_freq
         

def idf(document: Counter, text_corpus:list, basic_tfidf_vector):
    # length of text corpus
    n = len(text_corpus)
    # iterate through every word and count the documents
    # in which it also occures
    for word in document:
        count = 0
        for doc in text_corpus:
            if word in doc:
                count += 1
        basic_tfidf_vector[word] = np.log(n/count)
    return np.array(list(basic_tfidf_vector.values()))


def tfidf(document:Counter, corpus: list, basic_tfidf_vector: dict):
    base_vector = deepcopy(basic_tfidf_vector)
    # fill the basic tfidf vector
    # extract keys
    words = list(document.keys())
    # extract values
    for key, val in document.items():
        if key in base_vector.keys():
            base_vector[key] = val
    # turn into np vector
    counts_vector = np.array(list(base_vector.values()))
    # calculate tf vector of the document
    tf_vector = tf(counts_vector)
    # calculate idf vector of the document
    idf_vector = idf(document, corpus, base_vector)
    # calculate tfidf result
    result = tf_vector * idf_vector
    # return result and corresponding word list
    return result, words

def to_lower(document: str):
    return document.lower()

def remove_punctuation(document: str):
    punctuation = "?.,/|\\+*#':;!$%&§(){}¢[]'\"@=<-_>\n"
    translation_table = str.maketrans(punctuation, ' ' * len(punctuation))
    return document.translate(translation_table)

def remove_stop_words(document: str):
    stopwords = set(stopwords_en)
    result = ""
    for word in document.split():
        if word not in stopwords:
            result += word + " "
    return result

def convert_to_ascii(text):
    ascii_bytes = text.encode('ascii', 'ignore')
    ascii_text = ascii_bytes.decode('ascii')
    return ascii_text

def stem(document: str):
    ps = PorterStemmer()
    processed = ""
    for word in document.split():
        processed += ps.stem(word) + " "
    return processed

def to_counts(document: str) -> Counter:
    return Counter(document.split()) 

def preprocess_all(text_corpus : list) -> list:
    processed_corpus = []
    for document in text_corpus:
        processed_corpus.append(preprocess(document))
    return processed_corpus

def preprocess(document) -> Counter:
    document = to_lower(document)
    document = convert_to_ascii(document)
    document = remove_punctuation(document)
    document = remove_stop_words(document)
    document = stem(document)
    document = to_counts(document)
    return document

def get_basic_vector(text_corpus):
    basic_vector = {}
    for document in text_corpus:
        for word in document:
            if word not in basic_vector:
                basic_vector[word] = 0
    return basic_vector


def cosine_similarity(vector1, vector2):
    dot_pr = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    return dot_pr / (norm1 * norm2)
