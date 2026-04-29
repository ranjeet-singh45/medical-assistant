"""RAG pipeline module integrating retrieval and generation."""
from typing import Optional, Dict
from openai import OpenAI
import logging

from src.config import OPENAI_API_KEY, OPENAI_MODEL
from src.retriever import Retriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """End-to-end RAG pipeline."""
    
    def __init__(self, retriever: Retriever):
        """
        Initialize RAG Pipeline.
        
        Args:
            retriever: Retriever instance
        """
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = OPENAI_MODEL
        self.retriever = retriever
    
    def generate_answer(self, query: str, top_k: Optional[int] = None) -> Dict:
        """
        Generate answer using RAG pipeline.
        
        Args:
            query: User question
            top_k: Number of documents to retrieve
        
        Returns:
            Dictionary containing answer, context, and retrieved documents
        """
        # Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(query, top_k=top_k)
        
        # Format context
        context = self.retriever.format_context(retrieved_docs)
        
        # Create prompt
        prompt = f"""You are a medical assistant. Answer the user's question based on the provided context from medical documents. 
If the context doesn't contain enough information to answer the question, say so.

Context:
{context}

Question: {query}

Answer:"""
        
        # Generate answer using OpenAI
        try:
            logger.info("Generating answer using OpenAI...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful medical assistant that provides accurate information based on the given context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            answer = response.choices[0].message.content
            
            return {
                "answer": answer,
                "context": context,
                "retrieved_docs": retrieved_docs,
                "sources": [doc["metadata"] for doc in retrieved_docs]
            }
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise

