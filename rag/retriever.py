import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rag.embedder import build_index, MODEL_NAME, INDEX_PATH, CHUNKS_PATH

_state = {"index": None, "chunks": None, "model": None}


def _load():
    if _state["index"] is not None:
        return
    _state["model"] = SentenceTransformer(MODEL_NAME)
    if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
        _state["index"] = faiss.read_index(INDEX_PATH)
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            _state["chunks"] = json.load(f)
    else:
        idx, chunks, model = build_index()
        _state["index"] = idx
        _state["chunks"] = chunks
        _state["model"] = model


def retrieve(query, top_k=5):
    _load()
    emb = _state["model"].encode([query], normalize_embeddings=True).astype(np.float32)
    scores, indices = _state["index"].search(emb, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0:
            chunk = _state["chunks"][idx]
            results.append(
                {
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "score": float(score),
                }
            )
    return results


def rebuild_index():
    _state["index"] = None
    _state["chunks"] = None
    _state["model"] = None
    build_index()
    _load()
