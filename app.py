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
st.set_page_config(
    page_title="Smart Search Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #6c757d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .search-container {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        border: 1px solid #dee2e6;
    }
    .result-card {
        background-color: white;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: box-shadow 0.3s ease;
    }
    .result-card:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    .score-badge {
        background-color: #28a745;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .doc-content {
        margin-top: 0.5rem;
        color: #495057;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Search Settings")
    
    st.markdown("---")
    st.subheader("Results Configuration")
    TOP_K = st.slider("Number of results to show", min_value=1, max_value=20, value=5)
    THRESHOLD = st.slider("Similarity threshold", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
    
    st.markdown("---")
    st.subheader("Dataset Info")
    st.write(f"📄 Total documents: {len(documents)}")
    st.write(f"📝 Unique words: {len(vocab)}")
    
    st.markdown("---")
    st.subheader("About")
    st.write("This search engine uses TF-IDF and cosine similarity to find relevant documents.")

# Main content
st.markdown('<div class="main-header">🔍 Smart Search Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Discover relevant documents using advanced TF-IDF search technology</div>', unsafe_allow_html=True)
st.markdown("*Powered by manual TF-IDF implementation*")

# Search section
st.markdown('<div class="search-container">', unsafe_allow_html=True)
col1, col2 = st.columns([4, 1])

with col1:
    query = st.text_input("Enter your search query:", placeholder="e.g., machine learning, deep learning, artificial intelligence", help="Search for keywords or phrases related to AI, NLP, or machine learning topics.")

with col2:
    search_button = st.button("🔍 Search", type="primary", use_container_width=True)

st.caption("💡 Tip: Try specific terms like 'neural networks' or 'natural language processing' for better results.")
st.markdown('</div>', unsafe_allow_html=True)

# Search logic
if search_button and query:
    with st.spinner("Searching..."):
        query_tokens = preprocess(query)
        query_vec = compute_tfidf(query_tokens)
        
        candidates = get_candidate_docs(query_tokens)
        
        if not candidates:
            st.error("❌ No matching documents found. Try different keywords.")
        else:
            scores = [
                (i, cosine_sim(query_vec, doc_vectors[i]))
                for i in candidates
            ]
            
            scores.sort(key=lambda x: x[1], reverse=True)
            
            # Filter by threshold
            filtered_scores = [(i, score) for i, score in scores if score > THRESHOLD]
            
            if not filtered_scores:
                st.warning(f"⚠️ No results above the threshold of {THRESHOLD:.2f}. Try lowering the threshold or different query.")
            else:
                st.success(f"✅ Found {len(filtered_scores)} relevant documents")
                
                # Query info
                with st.expander("🔍 Query Analysis"):
                    st.write(f"**Original query:** {query}")
                    st.write(f"**Processed tokens:** {', '.join(query_tokens)}")
                    st.write(f"**Candidate documents:** {len(candidates)}")
                
                st.subheader("📋 Search Results")
                
                count = 0
                for i, score in filtered_scores:
                    if count >= TOP_K:
                        break
                    
                    with st.container():
                        st.markdown(f"""
                        <div class="result-card">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <h4 style="margin: 0; color: #1f77b4;">Document {i+1}</h4>
                                <span class="score-badge">Score: {score:.3f}</span>
                            </div>
                            <div class="doc-content">
                                {documents[i]}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    count += 1
                
                if len(filtered_scores) > TOP_K:
                    st.info(f"Showing top {TOP_K} results. {len(filtered_scores) - TOP_K} more results available.")

elif search_button and not query:
    st.warning("Please enter a search query.")

# Footer
st.markdown("---")
st.markdown("*Built with ❤️ using Streamlit and pure Python*")




