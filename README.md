# 🏗️ BIS Standards Recommendation Engine

> AI-powered BIS standard discovery for Indian Micro & Small Enterprises  
> **BIS × SS Hackathon · May 2026 · Track: AI / RAG**

---

## 📁 Repository Structure
your-repo/
│
├── inference.py          ← root 
├── eval_script.py        ← root 
├── requirements.txt      ← root
├── presentation.pdf      ← root
├── README.md             ← root
│
├── src/                  
│   ├── ingest.py
│   ├── rag_pipeline.py
│   ├── app.py
│   
│
└── data/                 
    └── public_test_results.json ← output of running inference.py on public test set

---

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/singhabhigyan007devil-cpu/bis-standards-rag.git
cd bis-standards-rag
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your Groq API key (only needed for the Streamlit app)

Create a file named `groq.env` in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

> > **Note:** `inference.py` does not require a Groq API key. It uses only BM25 + FAISS for retrieval with no LLM calls, making it fast and free to run.

---

## 🗃️ Step 1 — Build the Index (One-Time)

Place `dataset.pdf` in the project root, then run:
```bash
python src/ingest.py
```

This will:
- Extract and chunk all ~568 BIS standards from the PDF
- Embed each chunk using `all-MiniLM-L6-v2`
- Save `data/index.faiss` and `data/metadata.pkl`

Expected output:
```
[1/5] Extracting text ...   Characters: 1,784,677
[2/5] Splitting chunks ...  Raw chunks: 568
[3/5] Parsing metadata ...  Valid standards: 568 | Skipped: 0
[4/5] Embedding ...
[5/5] Building FAISS index ...
Done. 568 vectors saved to 'data/'
```

> ⚠️ You must run `ingest.py` before running any other script.

---

## 🤖 Step 2 — Run Inference (Judge Command)


```bash
python inference.py --input path/to/input.json --output path/to/output.json
```

For the hidden private test set (run by judges):

```bash
python inference.py --input hidden_private_dataset.json --output team_results.json
```

**Output format** (`team_results.json`):
```json
[
  {
    "id": "PUB-01",
    "retrieved_standards": ["IS 269 : 1989", "IS 455 : 1989", "IS 383 : 1970", "IS 1489 (PART1) : 1991", "IS 8112 : 1989"],
    "latency_seconds": 0.312,
    "expected_standards": ["IS 269: 1989"]
  },
  ...
]
```

---

## 📊 Step 3 — Evaluate Results

Run the organiser-provided evaluation script on your output:

```bash
python eval_script.py --results team_results.json
```

Expected output format:
```
========================================
   BIS HACKATHON EVALUATION RESULTS
========================================
Total Queries Evaluated : 10
Hit Rate @3             : 100.00%       (Target: >80%)
MRR @5                  : 0.8500        (Target: >0.7)
Avg Latency             : 0.06 sec      (Target: <5 seconds)
========================================
```

---

## 🌐 Step 4 — Run the Streamlit Demo (Optional)

```bash
python -m streamlit run src/app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

> Requires `GROQ_API_KEY` set in `groq.env`.

---

## 🔬 How It Works

### Pipeline Overview

```
User Query
   │
   ├─► BM25 Keyword Search  (top-25 candidates)
   │
   ├─► Dense Embedding Search via FAISS  (top-25 candidates, cosine similarity)
   │
   ├─► Score Fusion  (0.4 × BM25 + 0.6 × Dense)
   │
   └─► Top-5 Standards returned
```

### Key Design Decisions

| Component      | Choice                              | Why                                           |
|----------------|-------------------------------------|-----------------------------------------------|
| Chunking       | One chunk = one IS standard summary | Clean boundaries, no overlap needed           |
| PDF Parser     | PyMuPDF (fitz)                      | Better paragraph structure than pypdf         |
| Embedding      | `all-MiniLM-L6-v2`                  | Fast, accurate, free                          |
| Vector store   | FAISS `IndexFlatIP` (cosine)        | Exact search, deterministic                   |
| Keyword search | BM25 (custom, no external lib)      | Handles exact IS number lookups               |
| Fusion weights | 0.4 BM25 + 0.6 Dense                | Dense semantic wins, BM25 boosts exact matches|
| LLM (app only) | Groq Llama 3.1 8B                   | Free tier, <1s latency                        |

---

## 📦 Dependencies

```
pymupdf               # PDF parsing
sentence-transformers # Embedding model
faiss-cpu             # Vector search
numpy                 # Numerical ops
groq                  # LLM for rationale (app only)
streamlit             # Demo UI (app only)
python-dotenv         # API key loading
```

Install all with:
```bash
pip install -r requirements.txt 
```

---

## 🏆 Evaluation Targets

| Hit Rate @3 | > 80%   | **100%** |
| MRR @5      | > 0.70  | **0.85** |
| Avg Latency | < 5 sec | **0.08 sec** |

---

## 👥 Team

| Name           |                   Role                    |
|----------------|-------------------------------------------|
| Abhigyan Singh | RAG Pipeline, Evaluation & Data Ingestion |
| Waseem Raza    | UI/UX, Testing & Documentation            |

---

## 🙏 Acknowledgements

- **Bureau of Indian Standards (BIS)** — SP 21 dataset
- **Hugging Face** — `sentence-transformers` / `all-MiniLM-L6-v2`
- **Groq** — LLM inference platform
- **Organizers** — BIS × SS Hackathon