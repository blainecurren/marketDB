from typing import List, Dict, Optional, Any
from datetime import datetime
import hashlib
import json
from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct, Filter, FieldCondition, 
    MatchValue, Range, SearchParams
)
import numpy as np
from loguru import logger

class MarketDBVectorStore:
    """Wrapper for Qdrant operations specific to marketDB"""
    
    def __init__(self, host: str = "localhost", port: int = 6333):
        self.client = QdrantClient(host=host, port=port)
        self.collections = {
            "posts": "market_posts",
            "news": "market_news",
            "topics": "topic_summaries",
            "cache": "query_cache"
        }
        
    def add_market_posts(
        self, 
        posts: List[Dict], 
        embeddings: np.ndarray,
        symbol: str
    ) -> bool:
        """Add market posts with embeddings"""
        try:
            points = []
            for i, (post, embedding) in enumerate(zip(posts, embeddings)):
                # Generate unique ID
                post_id = hashlib.md5(
                    f"{post.get('id', i)}_{symbol}_{datetime.utcnow()}".encode()
                ).hexdigest()
                
                point = PointStruct(
                    id=post_id,
                    vector=embedding.tolist(),
                    payload={
                        "symbol": symbol,
                        "post_type": post.get("post_type", "unknown"),
                        "sentiment": post.get("sentiment", 3.0),
                        "interactions": post.get("interactions", 0),
                        "created_at": post.get("created_at", datetime.utcnow().isoformat()),
                        "post_id": str(post.get("id", "")),
                        "creator_name": post.get("creator_name", ""),
                        "text": post.get("text", "")[:1000],  # Limit text length
                        "indexed_at": datetime.utcnow().isoformat()
                    }
                )
                points.append(point)
            
            # Batch upsert
            self.client.upsert(
                collection_name=self.collections["posts"],
                points=points,
                wait=True
            )
            
            logger.info(f"Added {len(points)} posts for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add posts: {e}")
            return False
    
    def search_similar_posts(
        self,
        query_embedding: np.ndarray,
        symbol: Optional[str] = None,
        limit: int = 10,
        sentiment_min: Optional[float] = None
    ) -> List[Dict]:
        """Search for similar posts with optional filters"""
        
        # Build filter conditions
        conditions = []
        
        if symbol:
            conditions.append(
                FieldCondition(
                    key="symbol",
                    match=MatchValue(value=symbol)
                )
            )
        
        if sentiment_min:
            conditions.append(
                FieldCondition(
                    key="sentiment",
                    range=Range(gte=sentiment_min)
                )
            )
        
        # Create filter
        search_filter = Filter(must=conditions) if conditions else None
        
        # Search
        results = self.client.search(
            collection_name=self.collections["posts"],
            query_vector=query_embedding.tolist(),
            limit=limit,
            query_filter=search_filter,
            with_payload=True
        )
        
        return [
            {
                "score": hit.score,
                "payload": hit.payload
            }
            for hit in results
        ]
    
    def get_collection_stats(self, collection_type: str = "posts") -> Dict:
        """Get statistics for a collection"""
        collection_name = self.collections.get(collection_type)
        if not collection_name:
            return {"error": "Invalid collection type"}
        
        try:
            info = self.client.get_collection(collection_name)
            return {
                "name": collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status,
                "indexed_vectors": info.indexed_vectors_count
            }
        except Exception as e:
            return {"error": str(e)}