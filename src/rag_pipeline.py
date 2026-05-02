import re
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
import os

# Load .env file from the same directory as this script
load_dotenv(os.path.join(os.path.dirname(__file__), "groq.env"))
# ── Load saved index and chunks ───────────────────────────────────────
def load_index(data_dir="data/"):
    index = faiss.read_index(f"{data_dir}/index.faiss")
    with open("data/metadata.pkl", "rb") as f:  # ← this one actually exists
        metadata = pickle.load(f)
    return index, metadata

# ── Keyword scoring for reranking ────────────────────────────────────
def keyword_score(query, metadata):
    # Remove common stop words so technical terms get more weight
    stop_words = {'the', 'a', 'an', 'for', 'of', 'and', 'or', 'in', 'to',
                  'is', 'are', 'our', 'we', 'with', 'that', 'its', 'be',
                  'which', 'what', 'where', 'used', 'use', 'not', 'but',
                  'both', 'intended', 'looking', 'need', 'i', 'by', 'as'}
    query_words = set(query.lower().split()) - stop_words
    chunk_lower = metadata['text'].lower()
    score = sum(1 for word in query_words if word in chunk_lower)
    return score

# ── Extract standard number from chunk ───────────────────────────────
def extract_standard(metadata):
    match = re.search(r"IS\s+\d+(?:\s*\(Part\s+\d+\))?(?:\s*:\s*\d{4})?", metadata['text'])
    return match.group().strip() if match else "Unknown"

# ── Hybrid search: semantic + keyword reranking ───────────────────────
def search(query, model, index, metadata, k=5):
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding, dtype='float32'), k * 4)
    candidates = [(metadata[i], D[0][idx]) for idx, i in enumerate(I[0])]
    ranked = sorted(candidates, key=lambda x: keyword_score(query, x[0]), reverse=True)
    return [chunk for chunk, _ in ranked[:k]]

# ── LLM explanation via Groq ─────────────────────────────────────────
def generate_explanation(query, chunk, client):
    clean_chunk = re.sub(r"\s+", " ", chunk['text']).strip()[:800]
    
    prompt = f"""You are a BIS (Bureau of Indian Standards) compliance expert.

A user is looking for standards related to: "{query}"

Here is an excerpt from a relevant BIS standard:
{clean_chunk}

In 1-2 sentences, explain specifically WHY this standard is relevant to the user's product.
Only use information from the excerpt above. Do not make up standard numbers or content."""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=80,
        temperature=0.2
    )
    return response.choices[0].message.content.strip()

# ── Main pipeline function ────────────────────────────────────────────
def get_recommendations(query, model, index, metadata, groq_client, top_k=5):
    results = search(query, model, index, metadata, k=top_k)
    output = []
    seen_standards = set()
    
    for chunk in results:
        std = extract_standard(chunk)
        if std == "Unknown" or std in seen_standards:
            continue
        seen_standards.add(std)
        explanation = generate_explanation(query, chunk, groq_client)
        output.append({"standard": std, "reason": explanation})
        if len(output) >= 3:
            break
    
    return output

# ── Quick test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    
   
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "") # ← paste your key here
    
    print("Loading model and index...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    index, metadata = load_index()
    client = Groq(api_key=GROQ_API_KEY)
    
    query = "high strength cement for construction"
    results = get_recommendations(query, model, index, metadata, client)
    
    for i, r in enumerate(results, 1):
        print(f"\n{i}. {r['standard']}")
        print(f"   Reason: {r['reason']}")