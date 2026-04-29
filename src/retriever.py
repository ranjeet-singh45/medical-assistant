"""Retriever module for document retrieval."""
from typing import List, Dict, Optional
from src.vector_store import VectorStore
from src.config import TOP_K_RESULTS
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Retriever:
    """Handles retrieval of relevant documents."""
    
    def __init__(self, vector_store: VectorStore):
        """
        Initialize Retriever.
        
        Args:
            vector_store: VectorStore instance
        """
        self.vector_store = vector_store
        self.top_k = TOP_K_RESULTS
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve (defaults to config value)
        
        Returns:
            List of retrieved documents with metadata
        """
        if top_k is None:
            top_k = self.top_k
        
        logger.info(f"Retrieving top {top_k} documents for query: {query[:50]}...")
        results = self.vector_store.search(query, top_k=top_k)
        logger.info(f"Retrieved {len(results)} documents")
        
        return results
    
    def format_context(self, results: List[Dict]) -> str:
        """
        Format retrieved documents into context string.
        
        Args:
            results: List of retrieved document results
        
        Returns:
            Formatted context string
        """
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(f"Document {i}:\n{result['text']}\n")
        
        return "\n".join(context_parts)

