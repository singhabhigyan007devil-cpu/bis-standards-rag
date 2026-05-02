"""
inference.py — BIS RAG Inference Pipeline (Hybrid BM25 + Embedding)
=====================================================================
Run:  python inference.py --input public_test_set.json --output team_results.json
"""

import json, time, argparse, pickle, os, re, math
from collections import defaultdict
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

DATA_DIR   = "data/"
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K      = 5


# ── BM25 scorer ───────────────────────────────────────────────────────────────

def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1 = k1
        self.b  = b
        self.corpus_size = len(corpus)
        self.avgdl = sum(len(d) for d in corpus) / len(corpus)
        self.df = defaultdict(int)
        self.corpus = corpus
        for doc in corpus:
            for word in set(doc):
                self.df[word] += 1
        self.idf = {}
        for word, freq in self.df.items():
            self.idf[word] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1)

    def score(self, query_tokens, doc_idx):
        doc   = self.corpus[doc_idx]
        dl    = len(doc)
        score = 0.0
        tf_map = defaultdict(int)
        for w in doc:
            tf_map[w] += 1
        for word in query_tokens:
            if word not in self.idf:
                continue
            tf  = tf_map.get(word, 0)
            idf = self.idf[word]
            score += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return score

    def get_top_n(self, query_tokens, n=25):
        scores = [(self.score(query_tokens, i), i) for i in range(self.corpus_size)]
        scores.sort(reverse=True)
        return scores[:n]


# ── Load artifacts ─────────────────────────────────────────────────────────────

def load_artifacts(data_dir=DATA_DIR):
    index = faiss.read_index(os.path.join(data_dir, "index.faiss"))
    with open(os.path.join(data_dir, "metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)
    return index, metadata


# ── Hybrid retrieval ───────────────────────────────────────────────────────────

def retrieve(query, index, metadata, model, bm25, top_k=TOP_K):
    """
    1. BM25 keyword search  → top 25 candidates
    2. Dense embedding search → top 25 candidates
    3. Merge scores (normalised BM25 * 0.4 + cosine * 0.6)
    4. Return top_k deduplicated standard IDs
    """
    n_candidates = 25

    # ── BM25 ──
    qtoks = tokenize(query)
    bm25_hits = bm25.get_top_n(qtoks, n=n_candidates)
    max_bm25  = bm25_hits[0][0] if bm25_hits and bm25_hits[0][0] > 0 else 1.0
    bm25_scores = {metadata[i]["standard_id"]: s / max_bm25
                   for s, i in bm25_hits if s > 0}

    # ── Dense ──
    emb = model.encode([query]).astype("float32")
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10
    dists, indices = index.search(emb, n_candidates)
    max_cos = float(dists[0][0]) if dists[0][0] > 0 else 1.0
    dense_scores = {}
    for score, idx in zip(dists[0], indices[0]):
        if idx < 0: continue
        sid = metadata[idx]["standard_id"]
        dense_scores[sid] = float(score) / max_cos

    # ── Merge ──
    all_ids = set(bm25_scores) | set(dense_scores)
    combined = {}
    for sid in all_ids:
        b = bm25_scores.get(sid, 0.0)
        d = dense_scores.get(sid, 0.0)
        combined[sid] = 0.4 * b + 0.6 * d

    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    return [sid for sid, _ in ranked[:top_k]]


# ── Main ──────────────────────────────────────────────────────────────────────

def run(input_path, output_path):
    print(f"Loading model    : {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Loading index    : {DATA_DIR}")
    index, metadata = load_artifacts()
    print(f"Index size       : {index.ntotal} vectors")

    # Build BM25 corpus from metadata
    print("Building BM25 index ...")
    corpus = [tokenize(m["standard_id"] + " " + m["title"] + " " + m["text"])
              for m in metadata]
    bm25 = BM25(corpus)
    print(f"BM25 ready       : {len(corpus)} documents")

    with open(input_path, encoding="utf-8-sig") as f:
        queries = json.load(f)
    print(f"Queries loaded   : {len(queries)}\n")

    results = []
    for item in queries:
        qid   = item["id"]
        query = item["query"]

        t0        = time.time()
        retrieved = retrieve(query, index, metadata, model, bm25)
        latency   = round(time.time() - t0, 4)

        out = {
            "id":                  qid,
            "retrieved_standards": retrieved,
            "latency_seconds":     latency,
        }
        if "expected_standards" in item:
            out["expected_standards"] = item["expected_standards"]

        results.append(out)
        print(f"  [{qid}] {latency:.2f}s -> {retrieved}")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} results -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.input, args.output)