import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KB_DIR = os.path.join(BASE_DIR, "knowledge_base")
INDEX_PATH = os.path.join(BASE_DIR, "faiss_index.bin")
CHUNKS_PATH = os.path.join(BASE_DIR, "chunks.json")


def chunk_text(text, source, chunk_size=250, overlap=40):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append({"text": chunk, "source": source})
        i += chunk_size - overlap
        if i >= len(words):
            break
    return chunks


def build_index():
    model = SentenceTransformer(MODEL_NAME)
    all_chunks = []

    for fname in sorted(os.listdir(KB_DIR)):
        if not fname.endswith(".txt"):
            continue
        with open(os.path.join(KB_DIR, fname), "r", encoding="utf-8") as f:
            text = f.read()
        all_chunks.extend(chunk_text(text, fname))

    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    embeddings = embeddings.astype(np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f)

    return index, all_chunks, model
