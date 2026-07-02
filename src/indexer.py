import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Paths
CATALOG_PATH = os.path.join("data", "shl_product_catalog.json")
VECTOR_DIR = "vector_store"
INDEX_PATH = os.path.join(VECTOR_DIR, "shl_catalog.index")
METADATA_PATH = os.path.join(VECTOR_DIR, "metadata.json")

def build_index():
    print("Loading catalog data...")
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    print("Loading embedding model (this might take a minute the first time)...")
    # all-MiniLM-L6-v2 is fast, lightweight, and perfect for CPU-based FAISS
    model = SentenceTransformer('all-MiniLM-L6-v2')

    documents = []
    metadata = []

    print("Formatting documents for vectorization...")
    for item in catalog:
        # Create a rich text representation for the AI to search against
        name = item.get("name", "")
        desc = item.get("description", "")
        keys = item.get("keys", "")
        levels = item.get("job_levels", "")
        
        # Combine into a single searchable string
        search_text = f"Test Name: {name}. Type: {keys}. Job Levels: {levels}. Description: {desc}"
        documents.append(search_text)
        
        # Clean the list into a string to satisfy Pydantic schema validation
        raw_keys = item.get("keys", "Unknown")
        test_type_str = ", ".join(raw_keys) if isinstance(raw_keys, list) else str(raw_keys)

        # Store the clean metadata so we can return exactly what the API schema demands
        metadata.append({
            "name": name,
            "url": item.get("link", ""),
            "test_type": test_type_str,
            "languages": item.get("languages", ""),
            "duration": item.get("duration", ""),
            "description": desc
        })

    print(f"Generating embeddings for {len(documents)} assessments...")
    embeddings = model.encode(documents, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    os.makedirs(VECTOR_DIR, exist_ok=True)
    
    # Save the FAISS index
    faiss.write_index(index, INDEX_PATH)
    
    # Save the metadata mapping
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"Successfully indexed {len(documents)} items!")
    print(f"Index saved to {INDEX_PATH}")
    print(f"Metadata saved to {METADATA_PATH}")

if __name__ == "__main__":
    build_index()