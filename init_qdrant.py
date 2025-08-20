# init_qdrant.py
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient(host="localhost", port=6333)

# Define collections
collections_config = {
    "market_posts": {
        "size": 1024,
        "distance": Distance.COSINE
    },
    "market_news": {
        "size": 1024,
        "distance": Distance.COSINE
    },
    "topic_summaries": {
        "size": 1024,
        "distance": Distance.COSINE
    },
    "query_cache": {
        "size": 1024,
        "distance": Distance.COSINE
    }
}

print("Creating collections...")
for collection_name, config in collections_config.items():
    try:
        # Try to delete if exists
        client.delete_collection(collection_name)
        print(f"Deleted existing {collection_name}")
    except:
        pass
    
    # Create collection
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=config["size"],
            distance=config["distance"]
        )
    )
    print(f"✓ Created {collection_name}")

# Verify collections
collections = client.get_collections()
print("\nCollections created:")
for col in collections.collections:
    print(f"  - {col.name}")

print("\n✅ Qdrant is ready for data ingestion!")