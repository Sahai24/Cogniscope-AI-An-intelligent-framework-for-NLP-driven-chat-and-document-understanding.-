from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
from config import MILVUS_HOST, MILVUS_PORT, COLLECTION_NAME

# Constants
EMBEDDING_DIM = 1536  # ✅ Ensure this matches Azure OpenAI embeddings

# Connect to Milvus
connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)

def reset_milvus_collection():
    """Drops the existing collection and recreates it with the correct schema."""
    
    if COLLECTION_NAME in utility.list_collections():
        print(f"Dropping existing collection: {COLLECTION_NAME}")
        Collection(COLLECTION_NAME).drop()
    
    print(f"Creating new collection: {COLLECTION_NAME}")

    # Define schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),  # ✅ Auto-generated IDs
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4096)  # ✅ Store text chunks
    ]
    
    schema = CollectionSchema(fields, description="Document Embeddings")

    # Create collection
    collection = Collection(COLLECTION_NAME, schema)
    collection.create_index("embedding", {"metric_type": "COSINE"})  # 🔹 Similarity search index

    print(f"✅ Created new collection: {COLLECTION_NAME} with embedding dim={EMBEDDING_DIM}")
    return collection

# Initialize fresh collection
collection = reset_milvus_collection()

def store_embeddings(embeddings, texts):
    """Stores embeddings and corresponding text chunks in Milvus."""
    
    if not isinstance(embeddings, list) or not all(isinstance(e, list) for e in embeddings):
        raise ValueError("Embeddings should be a list of lists.")

    if any(len(e) != EMBEDDING_DIM for e in embeddings):
        raise ValueError(f"Each embedding must have {EMBEDDING_DIM} dimensions.")

    if len(embeddings) != len(texts):
        raise ValueError("Number of embeddings must match number of text chunks.")

    # ✅ Correct data format (Milvus expects only embedding & text, NOT IDs)
    data_to_insert = [
        embeddings,  # List of embeddings
        texts        # Corresponding text chunks
    ]

    collection.insert(data_to_insert)
    collection.flush()

    print(f"✅ Stored {len(embeddings)} embeddings with text in '{COLLECTION_NAME}'.")

def search_embeddings(query_embedding, top_k=3):
    """Performs similarity search in Milvus."""
    
    collection.load()
    
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    
    results = collection.search(
        data=[query_embedding],  # Query vector
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["text"]  # ✅ Retrieve text along with similarity score
    )
    
    retrieved_chunks = [hit.entity.get("text") for hit in results[0]]
    return retrieved_chunks
