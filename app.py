#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import streamlit as st
from collections import Counter, defaultdict
import nltk
from nltk.stem import PorterStemmer

# Download required data
nltk.download('punkt')

# -------------------------------
# 🔧 INITIAL SETUP
# -------------------------------
stemmer = PorterStemmer()

def preprocess(text):
    words = text.lower().split()
    return [stemmer.stem(word) for word in words]

# -------------------------------
# 📂 LOAD DATASET
# -------------------------------
def load_dataset(file):
    try:
        with open(file, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    except:
        # fallback dummy dataset
        base = [
            "deep learning improves neural networks",
            "machine learning and ai are related",
            "natural language processing is a part of ai",
            "deep neural networks are powerful",
            "ai applications include vision and nlp"
        ]
        return [base[i % len(base)] + f" {i}" for i in range(500)]

documents = load_dataset("dataset.txt")

# -------------------------------
# 🔤 PREPROCESS DOCUMENTS
# -------------------------------
tokenized_docs = [preprocess(doc) for doc in documents]

# Vocabulary
vocab = list(set(word for doc in tokenized_docs for word in doc))

# -------------------------------
# 📊 COMPUTE IDF
# -------------------------------
N = len(documents)
idf = {}

for word in vocab:
    df = sum(1 for doc in tokenized_docs if word in doc)
    idf[word] = math.log(N / (df + 1))

# -------------------------------
# 🧮 TF-IDF FUNCTION
# -------------------------------
def compute_tfidf(doc):
    tf = Counter(doc)
    doc_len = len(doc)
    tfidf = {}

    for word in vocab:
        tf_val = tf[word] / doc_len if word in tf else 0
        tfidf[word] = tf_val * idf[word]

    return tfidf

# Precompute document vectors
doc_vectors = [compute_tfidf(doc) for doc in tokenized_docs]

# -------------------------------
# ⚡ INVERTED INDEX
# -------------------------------
inverted_index = defaultdict(set)

for i, doc in enumerate(tokenized_docs):
    for word in doc:
        inverted_index[word].add(i)

def get_candidate_docs(query_tokens):
    candidates = set()
    for word in query_tokens:
        if word in inverted_index:
            candidates.update(inverted_index[word])
    return list(candidates)

# -------------------------------
# 📐 COSINE SIMILARITY
# -------------------------------
def cosine_sim(vec1, vec2):
    dot = sum(vec1[w] * vec2[w] for w in vocab)
    norm1 = math.sqrt(sum(vec1[w]**2 for w in vocab))
    norm2 = math.sqrt(sum(vec2[w]**2 for w in vocab))
    return dot / (norm1 * norm2 + 1e-9)

# -------------------------------
# 🌐 STREAMLIT UI
# -------------------------------
st.title("🔍 Smart Search Engine (Manual TF-IDF)")

query = st.text_input("Enter your query:")

TOP_K = 5
THRESHOLD = 0.1

if query:
    query_tokens = preprocess(query)
    query_vec = compute_tfidf(query_tokens)

    candidates = get_candidate_docs(query_tokens)

    if not candidates:
        st.warning("No matching documents found.")
    else:
        scores = [
            (i, cosine_sim(query_vec, doc_vectors[i]))
            for i in candidates
        ]

        scores.sort(key=lambda x: x[1], reverse=True)

        st.subheader("Top Results:")

        count = 0
        for i, score in scores:
            if score > THRESHOLD:
                st.write(f"Doc {i+1}: {documents[i]} → Score: {score:.2f}")
                count += 1
                if count == TOP_K:
                    break

        if count == 0:
            st.info("No results above threshold.")


# In[ ]:




