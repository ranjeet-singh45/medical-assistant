"""Vector store module for embeddings and similarity search."""
import os
import pickle
from typing import List, Dict, Optional
import numpy as np
from openai import OpenAI
import logging
from src.config import OPENAI_API_KEY, EMBEDDING_MODEL, VECTOR_STORE_PATH
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class VectorStore:
    """Manages vector embeddings and similarity search."""
   
    def __init__(self):
        """Initialize VectorStore with OpenAI client and load existing vector store if available."""
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
       
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.embedding_model = EMBEDDING_MODEL
        self.embeddings = []
        self.documents = []
        self.vector_store_path = VECTOR_STORE_PATH
        
        # Automatically load the vector store if it exists
        loaded = self.load_vector_store()
        if loaded:
            logger.info("Vector store loaded successfully on initialization.")
        else:
            logger.info("No existing vector store found; will create new one when embeddings are generated.")
   
    def create_embeddings(self, documents: List[Dict[str, str]]) -> None:
        """
        Create embeddings for documents.
       
        Args:
            documents: List of document dictionaries with 'text' and 'metadata' keys
        """
        logger.info(f"Creating embeddings for {len(documents)} documents...")
       
        texts = [doc["text"] for doc in documents]
       
        # Create embeddings in batches
        batch_size = 100
        all_embeddings = []
       
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            try:
                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=batch_texts
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                logger.info(f"Processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            except Exception as e:
                logger.error(f"Error creating embeddings for batch {i//batch_size + 1}: {e}")
                raise
       
        self.embeddings = np.array(all_embeddings)
        self.documents = documents
        logger.info("Embeddings created successfully")
        
        # Automatically save after creation
        self.save_vector_store()
   
    def save_vector_store(self) -> None:
        """Save vector store to disk."""
        try:
            data = {
                "embeddings": self.embeddings,
                "documents": self.documents
            }
            with open(self.vector_store_path, "wb") as f:
                pickle.dump(data, f)
            logger.info(f"Vector store saved to {self.vector_store_path}")
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")
            raise
   
    def load_vector_store(self) -> bool:
        """
        Load vector store from disk.
       
        Returns:
            True if loaded successfully, False otherwise
        """
        if not os.path.exists(self.vector_store_path):
            logger.info("Vector store not found, will create new one")
            return False
       
        try:
            with open(self.vector_store_path, "rb") as f:
                data = pickle.load(f)
            self.embeddings = data["embeddings"]
            self.documents = data["documents"]
            logger.info(f"Vector store loaded from {self.vector_store_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            return False
   
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Search for similar documents using cosine similarity.
       
        Args:
            query: Query text
            top_k: Number of top results to return
       
        Returns:
            List of similar documents with scores
        """
        if len(self.embeddings) == 0:
            raise ValueError("Vector store is empty. Please create embeddings first.")
       
        # Create query embedding
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=[query]
            )
            query_embedding = np.array(response.data[0].embedding)
        except Exception as e:
            logger.error(f"Error creating query embedding: {e}")
            raise
       
        # Calculate cosine similarity
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
       
        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
       
        # Return results with scores
        results = []
        for idx in top_indices:
            results.append({
                "document": self.documents[idx],
                "score": float(similarities[idx]),
                "text": self.documents[idx]["text"],
                "metadata": self.documents[idx]["metadata"]
            })
       
        return results