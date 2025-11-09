from pymilvus import connections, Collection
from config import MILVUS_HOST, MILVUS_PORT, COLLECTION_NAME

# Connect to Milvus
connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)

# Load Collection
collection = Collection(COLLECTION_NAME)
collection.load()

# Check Total Number of Embeddings
num_entities = collection.num_entities
print(f"✅ Total stored embeddings: {num_entities}")

# Fetch Sample Records
results = collection.query(
    expr="id >= 0",  
    output_fields=["id", "embedding", "text"],  
    limit=5  # Retrieve 5 records
)

print("\n✅ Sample Records:")
for doc in results:
    print(doc)  # Print ID, embedding, and text

# Check Embedding Dimensions
if results:
    embedding_sample = results[0]["embedding"]
    print(f"\n✅ Embedding shape: {len(embedding_sample)} (should be 1536)")
else:
    print("\n❌ No embeddings found in the collection!")
