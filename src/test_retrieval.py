import faiss
import json
import os
from sentence_transformers import SentenceTransformer

VECTOR_DIR = "vector_store"
INDEX_PATH = os.path.join(VECTOR_DIR, "shl_catalog.index")
METADATA_PATH = os.path.join(VECTOR_DIR, "metadata.json")

def test_embeddings(query_text, top_k=3):
    print(f"\n" + "="*50)
    print(f"Testing Query: '{query_text}'")
    print("="*50)
    
    # Load index and metadata
    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
        
    # We must use the exact same embedder model
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Generate the vector for our search query
    query_vector = embedder.encode([query_text]).astype("float32")
    
    # Perform the search (returns L2 distances and the array indices)
    distances, indices = index.search(query_vector, top_k)
    
    # Display the results
    for i, idx in enumerate(indices[0]):
        if idx != -1:
            match = metadata[idx]
            score = distances[0][i]
            print(f"\nResult {i+1} (L2 Distance: {score:.4f})")
            print(f"Name: {match['name']}")
            print(f"Type: {match['test_type']}")
            # Truncate description for readability
            desc = match['description']
            print(f"Desc: {desc[:150]}..." if len(desc) > 150 else f"Desc: {desc}")

if __name__ == "__main__":
    # Test Case 1: From the Full-Stack trace
    test_embeddings("Senior Full-Stack Engineer Java Spring REST API backend")
    
    # Test Case 2: From the Plant Operator trace
    test_embeddings("plant operators chemical facility safety reliability procedure compliance")