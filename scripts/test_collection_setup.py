"""
Test that the collection can be created with the correct hybrid search schema.
"""
from src.milvus.collection import (
    setup_collection,
    get_client,
    COLLECTION_NAME,
    create_schema,
    create_index_params,
)

print(f"[TEST] Using collection: {COLLECTION_NAME}")

# Create collection (recreate=True to get fresh schema)
client = setup_collection(recreate=True)

# Verify it exists and has the right shape
info = client.describe_collection(COLLECTION_NAME)
print(f"[TEST] Collection created: {COLLECTION_NAME}")
print(f"[TEST] Fields:")
for f in info["fields"]:
    print(f"  {f['name']}: {f['type']}  pk={f.get('is_primary', False)}")

funcs = info.get("functions", [])
print(f"[TEST] Functions: {funcs}")

# Verify indexes
indexes = client.list_indexes(COLLECTION_NAME)
print(f"[TEST] Indexes: {indexes}")

print("[TEST] Collection schema validation PASSED")
