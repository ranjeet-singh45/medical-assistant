"""Configuration module for the Medical RAG application."""
import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

# Data Configuration
DATA_PATH = os.getenv("DATA_PATH", "medDataset_processed.csv")
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "vector_store.pkl")

# RAG Configuration
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "3"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Chunk Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

