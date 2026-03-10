import os
import json
import numpy as np
import faiss
from datetime import datetime
from sentence_transformers import SentenceTransformer

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
MEMORY_FILE = os.path.join(DATA_DIR, "memory.json")
INDEX_FILE = os.path.join(DATA_DIR, "memory_index.bin")
VECTORS_FILE = os.path.join(DATA_DIR, "memory_vectors.npy")

os.makedirs(DATA_DIR, exist_ok=True)

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def load_all():
    if not os.path.exists(MEMORY_FILE):
        return []
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_all(records):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


def _rebuild_index(records):
    if not records:
        return
    model = _get_model()
    texts = [r.get("input_text", "") for r in records]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    embeddings = embeddings.astype(np.float32)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, INDEX_FILE)
    np.save(VECTORS_FILE, embeddings)


def store_interaction(input_text, input_type, parsed_problem, rag_context, solution, verification, feedback=None):
    records = load_all()
    record = {
        "id": len(records),
        "timestamp": datetime.now().isoformat(),
        "input_type": input_type,
        "input_text": input_text,
        "parsed_problem": parsed_problem,
        "rag_sources": [c["source"] for c in rag_context] if rag_context else [],
        "solution": solution,
        "verification": verification,
        "feedback": feedback,
    }
    records.append(record)
    _save_all(records)
    _rebuild_index(records)
    return record["id"]


def update_feedback(record_id, feedback):
    records = load_all()
    for r in records:
        if r["id"] == record_id:
            r["feedback"] = feedback
            break
    _save_all(records)


def find_similar(query, top_k=3, min_score=0.65):
    records = load_all()
    if not records or not os.path.exists(INDEX_FILE):
        return []

    model = _get_model()
    emb = model.encode([query], normalize_embeddings=True, show_progress_bar=False).astype(np.float32)

    index = faiss.read_index(INDEX_FILE)
    k = min(top_k, len(records))
    scores, indices = index.search(emb, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0 and float(score) >= min_score:
            r = dict(records[idx])
            r["similarity"] = float(score)
            results.append(r)
    return results


def get_ocr_corrections():
    records = load_all()
    corrections = {}
    for r in records:
        fb = r.get("feedback")
        if isinstance(fb, dict) and fb.get("type") == "ocr_correction":
            orig = fb.get("original", "")
            corrected = fb.get("corrected", "")
            if orig and corrected and orig != corrected:
                corrections[orig[:80]] = corrected[:80]
    return corrections
