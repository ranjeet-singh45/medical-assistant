"""Data loading and preprocessing module."""
import pandas as pd
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading and preprocessing of medical dataset."""
    
    def __init__(self, data_path: str):
        """
        Initialize DataLoader.
        
        Args:
            data_path: Path to the CSV file
        """
        self.data_path = data_path
        self.df = None
    
    def load_data(self) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Returns:
            DataFrame containing the medical Q&A data
        """
        try:
            logger.info(f"Loading data from {self.data_path}")
            self.df = pd.read_csv(self.data_path)
            logger.info(f"Loaded {len(self.df)} records")
            return self.df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def preprocess_data(self) -> List[Dict[str, str]]:
        """
        Preprocess data into documents for vector store.
        
        Returns:
            List of dictionaries containing document text and metadata
        """
        if self.df is None:
            self.load_data()
        
        documents = []
        for idx, row in self.df.iterrows():
            # Combine question and answer for better context
            text = f"Question: {row['Question']}\nAnswer: {row['Answer']}"
            
            doc = {
                "text": text,
                "metadata": {
                    "qtype": row.get("qtype", ""),
                    "question": row.get("Question", ""),
                    "answer": row.get("Answer", ""),
                    "index": idx
                }
            }
            documents.append(doc)
        
        logger.info(f"Preprocessed {len(documents)} documents")
        return documents
    
    def get_documents(self) -> List[Dict[str, str]]:
        """
        Get preprocessed documents.
        
        Returns:
            List of document dictionaries
        """
        return self.preprocess_data()

