"""
Quick verification script: check collection exists, show its schema/stats.
Uses only MilvusClient API (no old connections.py).
"""
from src.milvus.collection import get_client, COLLECTION_NAME

client = get_client()

print(f"=== Collection: {COLLECTION_NAME} ===")
print(f"Exists: {client.has_collection(COLLECTION_NAME)}")

if client.has_collection(COLLECTION_NAME):
    # Get collection info via MilvusClient
    stats = client.get_collection_stats(COLLECTION_NAME)
    print(f"Stats: {stats}")

    # Describe the schema
    coll_info = client.describe_collection(COLLECTION_NAME)
    print(f"Collection info: {coll_info}")

    # List indexes
    indexes = client.list_indexes(COLLECTION_NAME)
    print(f"Indexes: {indexes}")
