import asyncio
import httpx
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any
import hashlib
import json
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
import numpy as np
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

class LunarCrushIngestor:
    
    def __init__(self):
        # Initialize clients
        self.qdrant = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        self.encoder = SentenceTransformer(settings.EMBEDDING_MODEL)
        
        # API configuration
        self.headers = {
            "Authorization": f"Bearer {settings.LUNARCRUSH_API_KEY}"
        }
        
        # Cache for deduplication
        self.processed_posts = set()
        
        logger.info(f"Initialized LunarCrush ingestor")
        logger.info(f"Qdrant: {settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
        logger.info(f"Embedding model: {settings.EMBEDDING_MODEL}")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def fetch_coin_data(self, symbol: str) -> Optional[Dict]:
        """Fetch coin market data from LunarCrush"""
        async with httpx.AsyncClient() as client:
            try:
                url = f"{settings.LUNARCRUSH_BASE_URL}/public/coins/{symbol}/v1"
                response = await client.get(url, headers=self.headers, timeout=30.0)
                response.raise_for_status()
                
                data = response.json()
                logger.info(f"Fetched market data for {symbol}")
                return data.get("data", {})
                
            except httpx.HTTPError as e:
                logger.error(f"HTTP error fetching {symbol} data: {e}")
                return None
            except Exception as e:
                logger.error(f"Error fetching {symbol} data: {e}")
                return None
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def fetch_social_posts(self, symbol: str) -> List[Dict]:
        """Fetch social posts for a symbol"""
        async with httpx.AsyncClient() as client:
            try:
                # Use topic endpoint for social data
                topic = symbol.lower()
                url = f"{settings.LUNARCRUSH_BASE_URL}/public/topic/{topic}/posts/v1"
                
                response = await client.get(url, headers=self.headers, timeout=30.0)
                response.raise_for_status()
                
                data = response.json()
                posts = data.get("data", [])
                
                logger.info(f"Fetched {len(posts)} social posts for {symbol}")
                return posts[:settings.MAX_POSTS_PER_SYMBOL]  # Limit posts
                
            except httpx.HTTPError as e:
                logger.error(f"HTTP error fetching {symbol} posts: {e}")
                return []
            except Exception as e:
                logger.error(f"Error fetching {symbol} posts: {e}")
                return []
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def fetch_topic_summary(self, symbol: str) -> Optional[Dict]:
        """Fetch topic summary data"""
        async with httpx.AsyncClient() as client:
            try:
                topic = symbol.lower()
                url = f"{settings.LUNARCRUSH_BASE_URL}/public/topic/{topic}/v1"
                
                response = await client.get(url, headers=self.headers, timeout=30.0)
                response.raise_for_status()
                
                data = response.json()
                logger.info(f"Fetched topic summary for {symbol}")
                return data.get("data", {})
                
            except Exception as e:
                logger.error(f"Error fetching topic summary for {symbol}: {e}")
                return None
    
    def prepare_post_for_embedding(self, post: Dict, symbol: str) -> str:
        """Prepare post text for embedding"""
        # Combine relevant fields for rich embeddings
        parts = []
        
        # Add symbol context
        parts.append(f"Symbol: {symbol}")
        
        # Add post content
        if post.get("title"):
            parts.append(f"Title: {post['title']}")
        if post.get("body"):
            parts.append(f"Content: {post['body'][:500]}")  # Limit length
        
        # Add metadata for context
        if post.get("sentiment"):
            parts.append(f"Sentiment: {post['sentiment']}")
        if post.get("post_type"):
            parts.append(f"Type: {post['post_type']}")
            
        return " | ".join(parts)
    
    async def process_and_store_posts(self, posts: List[Dict], symbol: str, market_data: Optional[Dict] = None):
        """Process posts and store in Qdrant"""
        if not posts:
            return
        
        # Prepare texts for embedding
        texts = []
        payloads = []
        
        for post in posts:
            # Skip if already processed
            post_id = str(post.get("id", ""))
            if post_id in self.processed_posts:
                continue
            
            # Prepare text
            text = self.prepare_post_for_embedding(post, symbol)
            texts.append(text)
            
            # Prepare payload with market context
            payload = {
                # Post data
                "symbol": symbol,
                "post_id": post_id,
                "post_type": post.get("post_type", "unknown"),
                "title": post.get("title", ""),
                "body": (post.get("body", "") or "")[:1000],  # Limit stored text
                "sentiment": float(post.get("sentiment", 3.0)),
                "interactions": int(post.get("interactions", 0)),
                "creator_name": post.get("creator_name", ""),
                "post_created": post.get("created", ""),
                
                # Market context (if available)
                "market_price": market_data.get("price") if market_data else None,
                "market_cap": market_data.get("market_cap") if market_data else None,
                "percent_change_24h": market_data.get("percent_change_24h") if market_data else None,
                
                # Metadata
                "ingested_at": datetime.now(timezone.utc).isoformat(),
                "source": "lunarcrush",
                "embedding_text": text[:500]  # Store what was embedded
            }
            
            payloads.append(payload)
            self.processed_posts.add(post_id)
        
        if not texts:
            logger.info(f"No new posts to process for {symbol}")
            return
        
        # Generate embeddings in batches
        logger.info(f"Generating embeddings for {len(texts)} posts...")
        embeddings = self.encoder.encode(texts, batch_size=32, show_progress_bar=True)
        
        # Create points for Qdrant
        points = []
        for i, (embedding, payload) in enumerate(zip(embeddings, payloads)):
            point_id = str(uuid4())
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload=payload
                )
            )
        
        # Store in Qdrant
        try:
            self.qdrant.upsert(
                collection_name=settings.COLLECTION_POSTS,
                points=points,
                wait=True
            )
            logger.info(f"✅ Stored {len(points)} posts for {symbol}")
        except Exception as e:
            logger.error(f"Failed to store posts in Qdrant: {e}")
    
    async def process_topic_summary(self, symbol: str, topic_data: Dict, market_data: Optional[Dict] = None):
        """Process and store topic summary"""
        if not topic_data:
            return
        
        # Create summary text for embedding
        summary_parts = [
            f"Symbol: {symbol}",
            f"Topic Rank: {topic_data.get('topic_rank', 'N/A')}",
            f"24h Interactions: {topic_data.get('interactions_24h', 0)}",
            f"Contributors: {topic_data.get('num_contributors', 0)}",
            f"Posts: {topic_data.get('num_posts', 0)}"
        ]
        
        # Add sentiment breakdown if available
        sentiment_data = topic_data.get("types_sentiment", {})
        if sentiment_data:
            summary_parts.append(f"Sentiment by type: {json.dumps(sentiment_data)}")
        
        summary_text = " | ".join(summary_parts)
        
        # Generate embedding
        embedding = self.encoder.encode([summary_text])[0]
        
        # Create payload
        payload = {
            "topic": symbol.lower(),
            "symbol": symbol,
            "time_bucket": datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0).isoformat(),
            "summary_type": "hourly_snapshot",
            
            # Topic metrics
            "topic_rank": topic_data.get("topic_rank"),
            "interactions_24h": topic_data.get("interactions_24h", 0),
            "num_contributors": topic_data.get("num_contributors", 0),
            "num_posts": topic_data.get("num_posts", 0),
            "social_dominance": topic_data.get("social_dominance", 0),
            
            # Sentiment analysis
            "sentiment_breakdown": sentiment_data,
            "dominant_sentiment": self._calculate_dominant_sentiment(sentiment_data),
            
            # Market context
            "market_price": market_data.get("price") if market_data else None,
            "market_cap": market_data.get("market_cap") if market_data else None,
            
            # Metadata
            "ingested_at": datetime.now(timezone.utc).isoformat(),
            "summary_text": summary_text
        }
        
        # Store in Qdrant
        try:
            self.qdrant.upsert(
                collection_name=settings.COLLECTION_TOPICS,
                points=[
                    PointStruct(
                        id=str(uuid4()),
                        vector=embedding.tolist(),
                        payload=payload
                    )
                ],
                wait=True
            )
            logger.info(f"✅ Stored topic summary for {symbol}")
        except Exception as e:
            logger.error(f"Failed to store topic summary: {e}")
    
    def _calculate_dominant_sentiment(self, sentiment_data: Dict) -> float:
        """Calculate weighted average sentiment"""
        if not sentiment_data:
            return 3.0  # Neutral
        
        total_sentiment = 0
        total_weight = 0
        
        for platform, sentiment in sentiment_data.items():
            if isinstance(sentiment, (int, float)):
                # Simple average for now
                total_sentiment += sentiment
                total_weight += 1
        
        if total_weight > 0:
            # Convert from 0-100 to 1-5 scale
            avg_sentiment = (total_sentiment / total_weight) / 20 + 1
            return round(avg_sentiment, 2)
        
        return 3.0
    
    async def ingest_symbol(self, symbol: str):
        """Complete ingestion pipeline for a single symbol"""
        logger.info(f"🚀 Starting ingestion for {symbol}")
        
        # Fetch all data concurrently
        tasks = [
            self.fetch_coin_data(symbol),
            self.fetch_social_posts(symbol),
            self.fetch_topic_summary(symbol)
        ]
        
        market_data, posts, topic_data = await asyncio.gather(*tasks)
        
        # Process and store data
        if posts:
            await self.process_and_store_posts(posts, symbol, market_data)
        
        if topic_data:
            await self.process_topic_summary(symbol, topic_data, market_data)
        
        logger.info(f"✅ Completed ingestion for {symbol}")
    
    async def ingest_all_symbols(self, symbols: Optional[List[str]] = None):
        """Ingest data for multiple symbols"""
        symbols = symbols or settings.DEFAULT_SYMBOLS
        
        logger.info(f"Starting batch ingestion for {len(symbols)} symbols: {symbols}")
        
        # Process symbols concurrently but with some limit
        for i in range(0, len(symbols), 3):  # Process 3 at a time
            batch = symbols[i:i+3]
            tasks = [self.ingest_symbol(symbol) for symbol in batch]
            await asyncio.gather(*tasks)
            
            # Small delay between batches to avoid rate limiting
            if i + 3 < len(symbols):
                await asyncio.sleep(2)
        
        logger.info("✅ Batch ingestion complete")
    
    def get_ingestion_stats(self) -> Dict:
        """Get statistics about ingested data"""
        stats = {}
        
        for collection_name in [settings.COLLECTION_POSTS, settings.COLLECTION_TOPICS]:
            try:
                collection_info = self.qdrant.get_collection(collection_name)
                stats[collection_name] = {
                    "vectors_count": collection_info.vectors_count,
                    "points_count": collection_info.points_count,
                    "status": collection_info.status
                }
            except Exception as e:
                stats[collection_name] = {"error": str(e)}
        
        stats["processed_posts"] = len(self.processed_posts)
        
        return stats

async def main():
    """Test the ingestor with a single symbol"""
    ingestor = LunarCrushIngestor()
    
    # Test with one symbol
    await ingestor.ingest_symbol("BTC")
    
    # Print stats
    stats = ingestor.get_ingestion_stats()
    logger.info(f"Ingestion stats: {json.dumps(stats, indent=2)}")

if __name__ == "__main__":
    # Run test
    asyncio.run(main())