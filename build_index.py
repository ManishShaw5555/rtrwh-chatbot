# build_index.py
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path
import pickle

CHUNK_FILE = Path("chunks.json")
INDEX_FILE = Path("faiss_index.bin")
DOCS_META_FILE = Path("docs.json")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # lightweight and fast

def main():
    if not CHUNK_FILE.exists():
        raise FileNotFoundError("chunks.json not found. Run ingest.py first.")

    # Load chunks
    with CHUNK_FILE.open("r", encoding="utf-8") as f:
        chunks = json.load(f)

    documents = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    metas = [c.get("meta", {}) for c in chunks]

    print(f"Loaded {len(documents)} chunks.")

    # Load embedding model
    print(f"Loading embedding model: {EMBED_MODEL_NAME}")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    # Generate embeddings
    print("Generating embeddings...")
    embeddings = model.encode(documents, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype="float32")

    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Save index to disk
    faiss.write_index(index, str(INDEX_FILE))

    # Save metadata
    meta_data = {"documents": documents, "ids": ids, "metas": metas}
    with open(DOCS_META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta_data, f, ensure_ascii=False, indent=2)

    print(f"FAISS index built and saved to {INDEX_FILE}")
    print(f"Metadata saved to {DOCS_META_FILE}")

if __name__ == "__main__":
    main()
