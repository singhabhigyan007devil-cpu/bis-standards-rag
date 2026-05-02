"""
ingest.py — BIS SP 21 Ingestion Pipeline (Fixed v3)
=====================================================
Verified against dataset.pdf (SP 21:2005, 929 pages, ~568 standards).

FIXES vs v2:
  1. Key standard check: now uses word-boundary regex instead of naive substring
     `k in m["standard_id"]`. This eliminates false positives like IS 269
     matching IS 2691, and correctly detects IS 2185 / IS 6909 / IS 8112.

  2. Debug dump: when a key standard is MISSING, prints the 5 closest
     standard_ids in metadata so you can see the actual stored format.

  3. Docstring updated: 568 chunks (not 577) is correct for this PDF.
"""

import re
import os
import pickle
import numpy as np
import faiss
import fitz
from sentence_transformers import SentenceTransformer

SAVE_DIR   = "data/"
MODEL_NAME = "all-MiniLM-L6-v2"

# Handles all IS ID formats found in SP 21:2005
IS_ID_RE = re.compile(
    r"IS\s*:?\s*"                                # IS with optional colon
    r"(\d+)"                                     # standard number
    r"(?:\s*\(\s*PART\s*[\w\s/]+?\s*\))?"       # optional (PART ...) — flexible
    r"(?:"
        r"\s*[-:]\s*(\d{4})"                     # :YYYY or -YYYY
        r"|\s+(\d{4})\s*:?"                      # YYYY: or standalone YYYY
    r")",
    re.IGNORECASE
)

# Same pattern but MULTILINE so ^ anchors to line start — avoids body references
IS_ID_RE_LINE = re.compile(
    r"^IS\s*:?\s*"
    r"(\d+)"
    r"(?:\s*\(\s*PART\s*[\w\s/]+?\s*\))?"
    r"(?:"
        r"\s*[-:]\s*(\d{4})"
        r"|\s+(\d{4})\s*:?"
    r")",
    re.IGNORECASE | re.MULTILINE
)


def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text() for page in doc)


def split_chunks(text):
    """
    Split on 'SUMMARY OF'.
    Uses \\s+ between SUMMARY and OF to handle double-space variants
    that fitz renders on some pages (e.g. 'SUMMARY  OF').
    """
    parts = re.split(r"SUMMARY\s+OF\s*\n", text)
    chunks = []
    for part in parts[1:]:  # skip front matter
        cleaned = re.sub(r"\n{3,}", "\n\n", part).strip()
        if len(cleaned) > 80:
            chunks.append(cleaned)
    return chunks


def extract_id(chunk):
    """
    Extract the IS standard ID for this chunk.

    Uses line-start anchored regex (IS_ID_RE_LINE) to match only IDs
    that appear at the beginning of a line — the real standard ID is always
    on its own line right after 'SUMMARY OF\\n'. Body references like
    '...refer to IS 2116 : 1980...' are mid-sentence and won't match.

    Falls back to the unanchored regex if nothing found (edge cases).
    """
    m = IS_ID_RE_LINE.search(chunk)
    if m:
        return m.group(0).strip()
    m = IS_ID_RE.search(chunk)
    return m.group(0).strip() if m else None


def extract_title(chunk):
    """Extract title text that follows the IS ID on the first matching line."""
    lines = chunk.splitlines()
    title_parts = []
    for i, line in enumerate(lines[:10]):
        if IS_ID_RE.search(line):
            after = IS_ID_RE.sub("", line).strip()
            if after:
                title_parts.append(after)
            for j in range(i + 1, min(i + 3, len(lines))):
                nxt = lines[j].strip()
                if not nxt or re.match(r"^\d+\.", nxt) or nxt.startswith("("):
                    break
                title_parts.append(nxt)
            break
    return " ".join(title_parts).strip() or "Unknown"


def build_embed_text(std_id, title, body):
    """Enrich embedding text: ID + title + first 700 chars of body."""
    snippet = re.sub(r"\s+", " ", body[:700].replace("\n", " ")).strip()
    return f"BIS Standard {std_id} — {title}. {snippet}"


def key_check_hits(k, metadata):
    """
    FIX: Use word-boundary regex instead of substring `k in standard_id`.

    Problem with old approach:
      "IS 269" in "IS 2691 : 1988"  → True  (FALSE POSITIVE)
      "IS 269" in "IS 269 : 1989"   → True  (correct)

    Word-boundary approach:
      re.search(r"IS\s*269\b") on "IS 2691 : 1988" → no match (correct)
      re.search(r"IS\s*269\b") on "IS 269 : 1989"  → match   (correct)
    """
    # Extract the number part from key like "IS 269" → "269"
    num = re.search(r"\d+", k)
    if not num:
        return []
    # \b ensures we don't match IS 2691 when looking for IS 269
    pattern = re.compile(r"IS\s*:?\s*" + num.group() + r"\b", re.IGNORECASE)
    return [m["standard_id"] for m in metadata if pattern.search(m["standard_id"])]


def build_index(pdf_path, save_dir=SAVE_DIR):
    os.makedirs(save_dir, exist_ok=True)

    print(f"[1/5] Extracting text from {pdf_path} ...")
    text = extract_text(pdf_path)
    print(f"      Characters: {len(text):,}")

    print("[2/5] Splitting into per-standard chunks ...")
    raw = split_chunks(text)
    print(f"      Raw chunks: {len(raw)}  (expected ~568)")

    print("[3/5] Parsing metadata ...")
    metadata, skipped = [], 0
    skipped_samples = []
    for chunk in raw:
        std_id = extract_id(chunk)
        if not std_id:
            skipped += 1
            if len(skipped_samples) < 5:
                skipped_samples.append(chunk[:150])
            continue
        title = extract_title(chunk)
        metadata.append({
            "standard_id": std_id,
            "title":       title,
            "text":        chunk,
            "embed_text":  build_embed_text(std_id, title, chunk),
        })
    print(f"      Valid standards : {len(metadata)}")
    print(f"      Skipped         : {skipped}")

    if skipped_samples:
        print("\n      --- Still-skipped chunks (first 150 chars) ---")
        for i, s in enumerate(skipped_samples, 1):
            print(f"      [{i}] {repr(s)}\n")

    # ── Sanity check (fixed: word-boundary matching, debug on MISSING) ──
    KEY = ["IS 269", "IS 383", "IS 455", "IS 458", "IS 1489",
           "IS 2185", "IS 3466", "IS 6909", "IS 8042", "IS 8112"]
    print("\n      --- Key standard check ---")
    all_ids = [m["standard_id"] for m in metadata]
    for k in KEY:
        hits = key_check_hits(k, metadata)
        if hits:
            print(f"      {k}: OK  -> {hits}")
        else:
            print(f"      {k}: MISSING!")
            # Debug: show stored IDs that contain the number digits (loose search)
            num = re.search(r"\d+", k).group()
            near = [sid for sid in all_ids if num in sid][:5]
            if near:
                print(f"             ^ Closest stored IDs containing '{num}': {near}")
            else:
                print(f"             ^ No stored ID contains digits '{num}' at all.")
                print(f"               This standard may genuinely be absent from the PDF.")
    print()

    print(f"[4/5] Embedding with '{MODEL_NAME}' ...")
    model = SentenceTransformer(MODEL_NAME)
    embs  = model.encode(
        [m["embed_text"] for m in metadata],
        show_progress_bar=True,
        batch_size=64
    ).astype("float32")

    # Normalize -> cosine similarity
    embs /= np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10

    print("[5/5] Building FAISS index (cosine / IndexFlatIP) ...")
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)

    faiss.write_index(index, os.path.join(save_dir, "index.faiss"))
    with open(os.path.join(save_dir, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)

    print(f"\nDone. {index.ntotal} vectors saved to '{save_dir}'")
    return metadata, index


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidate1 = os.path.join(script_dir, "dataset.pdf")
        candidate2 = os.path.join(os.path.dirname(script_dir), "dataset.pdf")

        if os.path.exists(candidate1):
            pdf_path = candidate1
        elif os.path.exists(candidate2):
            pdf_path = candidate2
        else:
            print("ERROR: Could not find dataset.pdf automatically.")
            print(f"  Looked in: {candidate1}")
            print(f"  Looked in: {candidate2}")
            print("  Please place dataset.pdf in your project root or src/ folder.")
            sys.exit(1)

    print(f"Using dataset: {pdf_path}\n")
    build_index(pdf_path)