import os
import sys
from typing import Dict, List, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    OptimizersConfigDiff, CreateAliasOperation,
    CreateAlias
)
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class QdrantMarketDBInitializer:
    """Initialize Qdrant collections for market intelligence platform"""
    
    def __init__(self):
        self.host = os.getenv("QDRANT_HOST", "localhost")
        self.port = int(os.getenv("QDRANT_PORT", 6333))
        self.vector_dim = int(os.getenv("VECTOR_DIMENSION", 1536))
        
        # Try to connect
        try:
            self.client = QdrantClient(host=self.host, port=self.port)
            print(f"✅ Connected to Qdrant at {self.host}:{self.port}")
        except Exception as e:
            print(f"❌ Failed to connect to Qdrant: {e}")
            sys.exit(1)
    
    def get_collections_config(self) -> Dict:
        """Define all collections and their configurations"""
        return {
            # LunarCrush social posts
            os.getenv("QDRANT_COLLECTION_POSTS", "market_posts"): {
                "description": "Social media posts from LunarCrush with market sentiment",
                "indexed_fields": ["symbol", "post_type", "sentiment", "interactions", "created_at", "post_id", "creator_name"]
            },
            
            # News articles
            os.getenv("QDRANT_COLLECTION_NEWS", "market_news"): {
                "description": "News articles with embeddings for RAG",
                "indexed_fields": ["symbol", "source", "published_at", "sentiment"]
            },
            
            # Aggregated summaries
            os.getenv("QDRANT_COLLECTION_TOPICS", "topic_summaries"): {
                "description": "Aggregated topic summaries for time-based analysis",
                "indexed_fields": ["topic", "time_bucket", "summary_type", "dominant_sentiment"]
            },
            
            # Query cache
            os.getenv("QDRANT_COLLECTION_CACHE", "query_cache"): {
                "description": "Cached queries for performance optimization",
                "indexed_fields": ["query_hash", "timestamp", "hit_count"]
            }
        }
    
    def create_collection(self, name: str, config: Dict) -> bool:
        """Create a single collection with optimized settings"""
        try:
            # Check if exists
            collections = [col.name for col in self.client.get_collections().collections]
            if name in collections:
                print(f"ℹ️  Collection '{name}' already exists")
                return True
            
            print(f"\n🏗️  Creating collection: {name}")
            print(f"   Description: {config['description']}")
            
            # Create collection
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=self.vector_dim,
                    distance=Distance.COSINE,
                    on_disk=True  # Enable disk storage for large datasets
                ),
                optimizers_config=OptimizersConfigDiff(
                    indexing_threshold=20000,
                    memmap_threshold=50000,
                    default_segment_number=4,
                    max_segment_size=200000
                ),
                on_disk_payload=True
            )
            
            print(f"✅ Collection created successfully")
            
            # Create payload indexes for the fields
            for field in config.get("indexed_fields", []):
                try:
                    self.client.create_payload_index(
                        collection_name=name,
                        field_name=field,
                        field_schema="keyword"  # Works for most field types
                    )
                    print(f"   ✅ Created index: {field}")
                except Exception as e:
                    print(f"   ⚠️  Could not create index for {field}: {str(e)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error creating collection {name}: {e}")
            return False
    
    def create_aliases(self):
        """Create aliases for easy collection management"""
        aliases = {
            "posts_latest": os.getenv("QDRANT_COLLECTION_POSTS", "market_posts"),
            "news_latest": os.getenv("QDRANT_COLLECTION_NEWS", "market_news"),
            "topics_latest": os.getenv("QDRANT_COLLECTION_TOPICS", "topic_summaries")
        }
        
        print("\n🔗 Creating collection aliases...")
        
        for alias, collection in aliases.items():
            try:
                self.client.update_collection_aliases(
                    change_aliases_operations=[
                        CreateAliasOperation(
                            create_alias=CreateAlias(
                                collection_name=collection,
                                alias_name=alias
                            )
                        )
                    ]
                )
                print(f"✅ Alias created: {alias} → {collection}")
            except Exception as e:
                print(f"⚠️  Could not create alias {alias}: {e}")
    
    def insert_test_data(self):
        """Insert some test data to verify setup"""
        print("\n🧪 Inserting test data...")
        
        import numpy as np
        from datetime import datetime
        
        # Test data for market_posts
        test_posts = [
            {
                "id": "test_1",
                "symbol": "BTC",
                "post_type": "tweet",
                "sentiment": 4.2,
                "interactions": 1500,
                "created_at": datetime.utcnow().isoformat(),
                "post_id": "test_post_1",
                "creator_name": "crypto_analyst",
                "text": "Bitcoin showing strong support at 50k level"
            },
            {
                "id": "test_2",
                "symbol": "ETH",
                "post_type": "reddit",
                "sentiment": 3.8,
                "interactions": 890,
                "created_at": datetime.utcnow().isoformat(),
                "post_id": "test_post_2",
                "creator_name": "eth_developer",
                "text": "Ethereum L2 solutions gaining massive adoption"
            },
            {
                "id": "test_3",
                "symbol": "BTC",
                "post_type": "news",
                "sentiment": 4.5,
                "interactions": 5000,
                "created_at": datetime.utcnow().isoformat(),
                "post_id": "test_post_3",
                "creator_name": "crypto_news",
                "text": "Institutional investors increase Bitcoin allocation"
            }
        ]
        
        collection_name = os.getenv("QDRANT_COLLECTION_POSTS", "market_posts")
        
        points = []
        for i, post_data in enumerate(test_posts):
            # Generate random vector for testing
            vector = np.random.rand(self.vector_dim).tolist()
            
            point = PointStruct(
                id=i,
                vector=vector,
                payload=post_data
            )
            points.append(point)
        
        try:
            self.client.upsert(
                collection_name=collection_name,
                points=points
            )
            print(f"✅ Inserted {len(points)} test vectors into {collection_name}")
        except Exception as e:
            print(f"❌ Failed to insert test data: {e}")
    
    def verify_setup(self):
        """Verify all collections are properly configured"""
        print("\n🔍 Verifying marketDB Qdrant setup...")
        
        try:
            collections = self.client.get_collections().collections
            
            print(f"\n📊 Summary:")
            print(f"   Total collections: {len(collections)}")
            
            for col in collections:
                col_info = self.client.get_collection(col.name)
                print(f"\n   📁 {col.name}")
                print(f"      Status: {'✅' if col_info.status == 'green' else '⚠️ '} {col_info.status}")
                print(f"      Vectors: {col_info.vectors_count}")
                print(f"      Indexed: {col_info.indexed_vectors_count}")
                print(f"      Points: {col_info.points_count}")
                
                # Get config info
                config_info = col_info.config
                print(f"      Vector size: {config_info.params.vectors.size}")
                print(f"      Distance: {config_info.params.vectors.distance}")
                
        except Exception as e:
            print(f"❌ Verification failed: {e}")
    
    def test_search(self):
        """Test search functionality"""
        print("\n🔍 Testing search functionality...")
        
        import numpy as np
        
        try:
            collection_name = os.getenv("QDRANT_COLLECTION_POSTS", "market_posts")
            
            # Generate random query vector
            query_vector = np.random.rand(self.vector_dim).tolist()
            
            # Search without filter
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=3
            )
            
            print(f"\n   Search results (no filter):")
            for i, hit in enumerate(results):
                print(f"   {i+1}. Score: {hit.score:.4f}")
                print(f"      Symbol: {hit.payload.get('symbol', 'N/A')}")
                print(f"      Text: {hit.payload.get('text', 'N/A')[:50]}...")
            
            # Search with filter
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
            btc_filter = Filter(
                must=[
                    FieldCondition(
                        key="symbol",
                        match=MatchValue(value="BTC")
                    )
                ]
            )
            
            filtered_results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                query_filter=btc_filter,
                limit=3
            )
            
            print(f"\n   Search results (BTC only):")
            for i, hit in enumerate(filtered_results):
                print(f"   {i+1}. Score: {hit.score:.4f}")
                print(f"      Symbol: {hit.payload.get('symbol', 'N/A')}")
                print(f"      Text: {hit.payload.get('text', 'N/A')[:50]}...")
                
        except Exception as e:
            print(f"❌ Search test failed: {e}")
    
    def run(self):
        """Run the complete initialization"""
        print(f"\n🚀 Initializing Qdrant for marketDB")
        print("=" * 50)
        
        # Create collections
        collections = self.get_collections_config()
        success_count = 0
        
        for name, config in collections.items():
            if self.create_collection(name, config):
                success_count += 1
        
        print(f"\n✅ Created {success_count}/{len(collections)} collections")
        
        # Create aliases
        self.create_aliases()
        
        # Insert test data
        self.insert_test_data()
        
        # Verify setup
        self.verify_setup()
        
        # Test search
        self.test_search()
        
        print("\n✅ marketDB Qdrant initialization complete!")
        print(f"🌐 Dashboard available at: http://{self.host}:6333/dashboard")

def main():
    initializer = QdrantMarketDBInitializer()
    initializer.run()

if __name__ == "__main__":
    main()