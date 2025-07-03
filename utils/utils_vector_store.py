from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import logging
import sys
import os

# Add the project root to the Python path for standalone execution
current_dir_vs = os.path.dirname(os.path.abspath(__file__))
project_root_vs = os.path.abspath(os.path.join(current_dir_vs, '..', '..'))
if project_root_vs not in sys.path:
    sys.path.insert(0, project_root_vs)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VectorStore:
    """
    A wrapper class for a vector database (FAISS) to store and retrieve
    text embeddings.
    """
    def __init__(self, embedding_model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initializes the VectorStore with an embedding model and an empty FAISS index.

        Args:
            embedding_model_name (str): The name of the SentenceTransformers model to use.
        """
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension) # L2 distance for similarity
        self.metadata_store: List[Dict[str, Any]] = [] # To store metadata like original text and source
        logging.info(f"VectorStore initialized with model: {embedding_model_name}, dimension: {self.dimension}")

    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None):
        """
        Adds texts and their associated metadata to the vector store.
        Each text is converted into an embedding and added to the FAISS index.

        Args:
            texts (List[str]): A list of text chunks to add.
            metadatas (Optional[List[Dict[str, Any]]]): A list of dictionaries,
                                                        where each dictionary contains
                                                        metadata for the corresponding text.
                                                        Must be same length as texts.
        """
        if not texts:
            return

        logging.info(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.embedding_model.encode(texts).astype(np.float32)
        logging.info("Embeddings generated. Adding to FAISS index.")

        # If no metadatas are provided, create default ones
        if metadatas is None:
            metadatas = [{"text": text} for text in texts]
        elif len(metadatas) != len(texts):
            raise ValueError("Length of metadatas must match length of texts.")

        self.index.add(embeddings)
        self.metadata_store.extend(metadatas)
        logging.info(f"Added {len(texts)} texts to vector store. Total: {self.index.ntotal}")

    def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Performs a semantic search for the most similar text chunks to the given query.

        Args:
            query (str): The user's query string.
            k (int): The number of top relevant chunks to retrieve.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary
                                  contains the 'text', 'source' (if available),
                                  and 'similarity_score' of a retrieved chunk.
                                  Sorted by similarity score (highest first).
        """
        if self.index.ntotal == 0:
            logging.warning("Vector store is empty. No search can be performed.")
            return []

        query_embedding = self.embedding_model.encode([query]).astype(np.float32)

        # D, I are distances and indices
        distances, indices = self.index.search(query_embedding, k)

        results = []
        for i, dist in zip(indices[0], distances[0]):
            if i == -1: # FAISS returns -1 for not found indices if k > total
                continue
            metadata = self.metadata_store[i]
            # FAISS returns L2 distance. Smaller distance means more similar.
            # Convert to a similarity score (higher is better) for clarity.
            # A common way is to use 1 / (1 + distance), or just report distance.
            # For L2, smaller is better, so we'll just return the distance as score for now.
            # You might want to normalize this for a '0-1' score if needed.
            similarity_score = 1.0 / (1.0 + dist) # Simple inverse mapping
            results.append({
                "text": metadata.get("text", "No text available"),
                "source": metadata.get("source", "Unknown"),
                "similarity_score": similarity_score
            })
        # Sort by similarity score in descending order
        return sorted(results, key=lambda x: x["similarity_score"], reverse=True)

    def clear_store(self):
        """Clears all data from the vector store."""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata_store = []
        logging.info("Vector store cleared.")

# Example Usage
if __name__ == "__main__":
    vector_store = VectorStore()

    texts_to_add = [
        "The quick brown fox jumps over the lazy dog.",
        "A dog is a loyal companion.",
        "Foxes are known for their cunning.",
        "Cats often sleep in sunny spots.",
        "The red car sped down the highway.",
        "KPIs are essential for business performance measurement."
    ]
    metadatas_to_add = [
        {"source": "story.txt", "page": 1, "text": texts_to_add[0]},
        {"source": "pet_guide.pdf", "section": "dogs", "text": texts_to_add[1]},
        {"source": "wildlife.md", "category": "mammals", "text": texts_to_add[2]},
        {"source": "pet_guide.pdf", "section": "cats", "text": texts_to_add[3]},
        {"source": "news.txt", "topic": "traffic", "text": texts_to_add[4]},
        {"source": "business_report.pdf", "chapter": "metrics", "text": texts_to_add[5]}
    ]

    vector_store.add_texts(texts_to_add, metadatas_to_add)

    query = "What measures business success?"
    results = vector_store.similarity_search(query, k=2)

    print(f"\nSearch results for '{query}':")
    for res in results:
        print(f"  - Text: '{res['text'][:50]}...'")
        print(f"    Source: {res['source']}, Score: {res['similarity_score']:.4f}")

    query_animals = "animals that are clever"
    results_animals = vector_store.similarity_search(query_animals, k=1)
    print(f"\nSearch results for '{query_animals}':")
    for res in results_animals:
        print(f"  - Text: '{res['text']}'")
        print(f"    Source: {res['source']}, Score: {res['similarity_score']:.4f}")

    vector_store.clear_store()
    print(f"\nAfter clearing, total items: {vector_store.index.ntotal}")