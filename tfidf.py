from collections import Counter
from copy import deepcopy
import numpy as np
from stopwords import stopwords_en
from nltk.stem import PorterStemmer
import os

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
    # Konvertiere den Text in Bytes mit ASCII-Encoding, ersetze nicht-ASCII-Zeichen mit einem Fragezeichen
    ascii_bytes = text.encode('ascii', 'ignore')
    # Konvertiere die Bytes zurück in einen String
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

def read_files(path: str):
    text_data = []
    files = os.listdir(path)
    for file in files:
        # Voller Pfad zur Datei
        file_path = os.path.join(path, file)
        # Überprüfen, ob es sich um eine Textdatei handelt
        if os.path.isfile(file_path) and file.endswith('.txt'):
            # Öffne die Datei im Lesemodus
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                text_data.append(content)
    return text_data

def cosine_similarity(vector1, vector2):
    # calculate the dot product
    dot_pr = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    return dot_pr / (norm1 * norm2)


if __name__ == "__main__":
    test_korpus = [
            "The CAT layed on the mat and the mat was green @ 5 AM",
            "The CAT layed on the mat and the mat was green @ 5 AM",
    "The CAT sat on the mat and the mat was green @ 3 PM",
    "A Dog layed on the grass and grass was purple sun",
    "The dog chased the CAT, but the cat climbed the tree!!!",
    "Birds flew over the TREE, and the DOG watched the bird...",
    "The fish swam in the pond while the bird sat on the TREE at 5 PM."]

    test_korpus = preprocess(test_korpus)
    basic_tfidf_vector = get_basic_vector(test_korpus)

    tfidf_matrix = []
    for doc in test_korpus:
        tfidf_vector, words = tfidf(doc,test_korpus, basic_tfidf_vector)
        tfidf_matrix.append(tfidf_vector)
              
    print(cosine_similarity(tfidf_matrix[0], tfidf_matrix[1]))
    print(cosine_similarity(tfidf_matrix[1], tfidf_matrix[2]))
    print(cosine_similarity(tfidf_matrix[2], tfidf_matrix[3]))


    # our text corpus
    #text_corpus = read_files('./corpus')
    # process our texts
    #text_corpus = preprocess(text_corpus)
    #basic_tfidf_vector = get_basic_vector(text_corpus)

    # transform documents into tfidf-vectors
    #for document in text_corpus:
        #result, words = tfidf(document, text_corpus, basic_tfidf_vector)
    

