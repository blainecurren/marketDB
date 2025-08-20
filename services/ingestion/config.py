# services/ingestion/config.py
import os
from typing import List
from dotenv import load_dotenv

load_dotenv()

# Try to import from pydantic v2 first, fall back to v1
try:
    from pydantic_settings import BaseSettings
    PYDANTIC_V2 = True
except ImportError:
    from pydantic import BaseSettings
    PYDANTIC_V2 = False

class Settings(BaseSettings):
    # LunarCrush API
    LUNARCRUSH_API_KEY: str = os.getenv("LUNARCRUSH_API_KEY", "")
    LUNARCRUSH_BASE_URL: str = "https://lunarcrush.com/api4"
    
    # Qdrant
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", 6333))
    
    # Collections
    COLLECTION_POSTS: str = "market_posts"
    COLLECTION_NEWS: str = "market_news"
    COLLECTION_TOPICS: str = "topic_summaries"
    
    # Embedding Model
    EMBEDDING_MODEL: str = "BAAI/bge-large-en-v1.5"
    
    # Ingestion Config
    INGESTION_INTERVAL: int = int(os.getenv("INGESTION_INTERVAL", 1))  # in hours
    
    # Ingestion settings
    BATCH_SIZE: int = 50
    MAX_POSTS_PER_SYMBOL: int = 100
    UPDATE_INTERVAL_MINUTES: int = 15
    
    # Symbols to track
    DEFAULT_SYMBOLS: List[str] = ["BTC", "ETH", "SOL", "AVAX", "MATIC"]
    
    @property
    def SYMBOLS(self) -> List[str]:
        """Get symbols from environment or use defaults"""
        symbols_env = os.getenv("SYMBOLS", "")
        if symbols_env:
            return [s.strip() for s in symbols_env.split(",") if s.strip()]
        return self.DEFAULT_SYMBOLS
    
    class Config:
        env_file = ".env"
        if PYDANTIC_V2:
            extra = "ignore"

settings = Settings()