# fix_qdrant_dimensions.py
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import time

print("Fixing Qdrant collections dimension mismatch...")

client = QdrantClient(host="localhost", port=6333)

# Step 1: Delete ALL collections
print("\nCurrent collections:")
collections = client.get_collections().collections
for col in collections:
    print(f"  - {col.name}")

print("\nDeleting all collections...")
for col in collections:
    try:
        client.delete_collection(col.name)
        print(f"  Deleted {col.name}")
        time.sleep(0.5)
    except Exception as e:
        print(f"  Error deleting {col.name}: {e}")

# Step 2: Verify all deleted
time.sleep(2)
remaining = client.get_collections().collections
if remaining:
    print(f"\nStill have collections: {[c.name for c in remaining]}")
else:
    print("\nAll collections deleted")

# Step 3: Recreate with correct dimension
print("\nCreating new collections with 1024 dimensions...")

collections_config = {
    "market_posts": "Social media posts",
    "market_news": "News articles", 
    "topic_summaries": "Aggregated summaries",
    "query_cache": "Search cache"
}

for name, description in collections_config.items():
    try:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=1024,
                distance=Distance.COSINE,
                on_disk=True
            )
        )
        print(f"  Created {name} (1024 dims) - {description}")
        time.sleep(0.5)
    except Exception as e:
        print(f"  Error creating {name}: {e}")

# Step 4: Verify
print("\nVerifying new collections...")
new_collections = client.get_collections().collections
for col in new_collections:
    info = client.get_collection(col.name)
    dim = info.config.params.vectors.size
    if dim == 1024:
        print(f"  - {col.name}: {dim} dimensions OK")
    else:
        print(f"  - {col.name}: {dim} dimensions WRONG")

print("\nFix complete! Your collections now expect 1024-dimensional vectors.")