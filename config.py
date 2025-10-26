from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    GEMINI_MODEL: str = "gemini-1.5-flash"
    
    VECTOR_DIMENSION: int = 384
    FAISS_INDEX_PATH: str = "./faiss_index"
    METADATA_PATH: str = "./metadata.json"
    
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_FILE_SIZE: int = 50 * 1024 * 1024
    
    DEFAULT_TOP_K: int = 5
    MAX_CONTEXT_LENGTH: int = 4000
    TEMPERATURE: float = 0.3
    
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()

if not settings.GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY must be set in environment variables or .env file")