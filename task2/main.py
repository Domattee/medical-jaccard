import string
import time

import pandas as pd
import numpy as np
from datasketch import MinHash
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import os

DATA_PATH = os.path.abspath(os.path.join(__file__, "..", "..", "data"))
DOC_PATH = os.path.join(DATA_PATH, "mtsamples.csv")
STOPWORD_PATH = os.path.join(DATA_PATH, "clinical-stopwords.txt")

LANGUAGE = "english"


# TODO: Also include description in shingles, but do not shingle end of description with start of transcription


try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download("punkt")


# Load datasets
def load_dataset(path=DOC_PATH):
    df = pd.read_csv(path, header=0)
    df = df.dropna(subset=["transcription"])
    df = df.reset_index()
    return df


def load_stopwords(path=STOPWORD_PATH):
    stopwords = set()
    with open(path, "rt", encoding="utf8") as f:
        header = f.readline()
        for line in f:
            stopwords.add(line.strip())
    return stopwords


# Select documents
def select_by_id(df, idx):
    return df["transcription"][idx]


def select_by_keyword():
    pass


def select_by_specialty():
    pass


# Process datasets, tokenize, etc.
def _tokenize_text(text):
    sents = sent_tokenize(text, LANGUAGE)
    words = [word for sent in sents for word in word_tokenize(sent, LANGUAGE) if word not in string.punctuation]
    return words


# Remove filter words
def _filter_words(wordlist, filterwords):
    filtered = [word for word in wordlist if word not in filterwords]
    return filtered


# Make shingles for single document
def _k_shingles(wordlist, k=2):
    shingles = set()
    for i in range(len(wordlist)-(k-1)):
        shingle = " ".join(wordlist[i:i+k])
        shingles.add(shingle)
    return shingles


# Complete processing pipe until shingle output
def build_k_shingles(text, k=2, filterwords=None):
    tokens = _tokenize_text(text)
    if filterwords:
        tokens = _filter_words(tokens, filterwords)
    shingles = _k_shingles(tokens, k)
    return shingles


def characteristic_matrix(df, k, filterwords):
    # Get all the shingles

    shingles = df["transcription"].apply(build_k_shingles, k=k, filterwords=filterwords)
    document_shingles = shingles.tolist()

    # Get set of all shingles
    set_of_all_shingles = set()
    for document_shingle in document_shingles:
        set_of_all_shingles.update(document_shingle)
    all_shingles = sorted(list(set_of_all_shingles))
    shingle_dict = {}
    for i, shingle in enumerate(all_shingles):
        shingle_dict[shingle] = i

    num_docs = len(document_shingles)
    num_shingles = len(set_of_all_shingles)

    matrix = np.zeros((num_shingles, num_docs))
    for j, document_shingle in enumerate(document_shingles):
        for word_shingle in document_shingle:
            i = shingle_dict[word_shingle]
            matrix[i, j] = 1

    return matrix


# Compute jaccard and minhash for documents
def jaccard(shingles1: set, shingles2: set):
    return float(len(shingles1.intersection(shingles2)))/float(len(shingles1.union(shingles2)))


def min_hash(shingles1: set, shingles2: set):
    m1 = MinHash()
    m2 = MinHash()
    m1.update_batch([shingle.encode("utf8") for shingle in shingles1])
    m2.update_batch([shingle.encode("utf8") for shingle in shingles2])
    return m1.jaccard(m2)


if __name__ == "__main__":
    df = load_dataset()
    stopwords = load_stopwords()
    shingles_a = build_k_shingles(df["transcription"][0], 2, filterwords=stopwords)
    shingles_b = build_k_shingles(df["transcription"][1], 2, filterwords=stopwords)
    shingles_c = build_k_shingles(df["transcription"][2], 2, filterwords=stopwords)

    start = time.time()
    c_matrix = characteristic_matrix(df, k=3, filterwords=stopwords)
    print(time.time() - start)
