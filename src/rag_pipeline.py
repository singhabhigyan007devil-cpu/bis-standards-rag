import re
import math
import numpy as np
import faiss
import pickle
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "groq.env"), override=True)

# ── Load index ────────────────────────────────────────────────────────
def load_index(data_dir="data/"):
    index = faiss.read_index(os.path.join(data_dir, "index.faiss"))
    with open(os.path.join(data_dir, "metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

# ── BM25 ──────────────────────────────────────────────────────────────
def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.avgdl = sum(len(d) for d in corpus) / len(corpus)
        self.df = defaultdict(int)
        self.corpus = corpus
        for doc in corpus:
            for word in set(doc):
                self.df[word] += 1
        self.idf = {}
        for word, freq in self.df.items():
            self.idf[word] = math.log(
                (self.corpus_size - freq + 0.5) / (freq + 0.5) + 1)

    def score(self, query_tokens, doc_idx):
        doc = self.corpus[doc_idx]
        dl = len(doc)
        score = 0.0
        tf_map = defaultdict(int)
        for w in doc:
            tf_map[w] += 1
        for word in query_tokens:
            if word not in self.idf:
                continue
            tf = tf_map.get(word, 0)
            idf = self.idf[word]
            score += idf * (tf * (self.k1 + 1)) / (
                tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return score

    def get_top_n(self, query_tokens, n=25):
        scores = [(self.score(query_tokens, i), i)
                  for i in range(self.corpus_size)]
        scores.sort(reverse=True)
        return scores[:n]

# ── Build BM25 from metadata ──────────────────────────────────────────
def build_bm25(metadata):
    corpus = [tokenize(m["standard_id"] + " " + m["title"] + " " + m["text"])
              for m in metadata]
    return BM25(corpus)

# ── Hybrid retrieval (same as inference.py) ───────────────────────────
def retrieve(query, model, index, metadata, bm25, top_k=5):
    n_candidates = 25
    qtoks = tokenize(query)

    # BM25
    bm25_hits = bm25.get_top_n(qtoks, n=n_candidates)
    max_bm25 = bm25_hits[0][0] if bm25_hits and bm25_hits[0][0] > 0 else 1.0
    bm25_scores = {metadata[i]["standard_id"]: s / max_bm25
                   for s, i in bm25_hits if s > 0}

    # Dense
    emb = model.encode([query]).astype("float32")
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10
    dists, indices = index.search(emb, n_candidates)
    max_cos = float(dists[0][0]) if dists[0][0] > 0 else 1.0
    dense_scores = {}
    for score, idx in zip(dists[0], indices[0]):
        if idx < 0:
            continue
        sid = metadata[idx]["standard_id"]
        dense_scores[sid] = float(score) / max_cos

    # Merge
    all_ids = set(bm25_scores) | set(dense_scores)
    combined = {}
    for sid in all_ids:
        b = bm25_scores.get(sid, 0.0)
        d = dense_scores.get(sid, 0.0)
        combined[sid] = 0.4 * b + 0.6 * d

    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    return [sid for sid, _ in ranked[:top_k]]

# ── LLM explanation ───────────────────────────────────────────────────
def generate_explanation(query, chunk_text, client):
    clean = re.sub(r"\s+", " ", chunk_text).strip()[:800]
    prompt = f"""You are a BIS compliance expert.
User query: "{query}"
BIS standard excerpt: {clean}
In 1-2 sentences, explain WHY this standard is relevant. Only use the excerpt."""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=80,
        temperature=0.2
    )
    return response.choices[0].message.content.strip()

# ── Main function for Streamlit app ──────────────────────────────────
def get_recommendations(query, model, index, metadata, groq_client, top_k=5):
    bm25 = build_bm25(metadata)
    retrieved_ids = retrieve(query, model, index, metadata, bm25, top_k)

    output = []
    meta_dict = {m["standard_id"]: m for m in metadata}

    for sid in retrieved_ids:
        if sid not in meta_dict:
            continue
        chunk_text = meta_dict[sid]["text"]
        explanation = generate_explanation(query, chunk_text, groq_client)
        output.append({"standard": sid, "reason": explanation})
        if len(output) >= 3:
            break

    return output