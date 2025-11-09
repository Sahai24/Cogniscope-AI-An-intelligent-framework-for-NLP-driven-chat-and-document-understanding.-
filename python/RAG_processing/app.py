import os
import shutil
from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields
from dotenv import load_dotenv
from openai import AzureOpenAI
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
from config import (
    DATA_INPUT_FOLDER, PROCESSED_FOLDER,
    AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT_EMBEDDING,
    DEPLOYMENT_CHAT, AZURE_OPENAI_API_VERSION,
    MILVUS_HOST, MILVUS_PORT, COLLECTION_NAME
)

# Load environment variables
load_dotenv()

# Initialize Flask App
app = Flask(__name__)
api = Api(app, title="Document Processing API", description="API for PDF processing, embedding, and Milvus storage", version="1.0")

# Define API namespaces
ns_processing = api.namespace("documents_processing", description="Operations related to document processing")
ns_query = api.namespace("documents_query", description="Operations related to querying documents")

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION
)

# Connect to Milvus
connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)

# Define query model for Swagger UI
query_model = api.model("QueryModel", {
    "query": fields.String(required=True, description="User query in JSON format")
})

# Ensure folders exist
os.makedirs(DATA_INPUT_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Define Constants
EMBEDDING_DIM = 1536  # Must match Azure OpenAI embeddings


# ✅ **Function: Extract text from PDF**
def extract_text_from_pdf(file_path):
    import pypdf
    """Extract text from each page of a PDF."""
    chunk_text = []
    with open(file_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                chunk_text.append(text.strip())  # Each page is one chunk
    return chunk_text


# ✅ **Function: Generate Embeddings**
def embed_text(chunks):
    """Generate embeddings using Azure OpenAI."""
    try:
        response = client.embeddings.create(input=chunks, model=AZURE_OPENAI_DEPLOYMENT_EMBEDDING)
        if hasattr(response, "data") and isinstance(response.data, list):
            return [item.embedding for item in response.data]  # Extract embeddings
    except Exception as e:
        print(f"Embedding Error: {e}")
    raise ValueError("Failed to generate embeddings with Azure OpenAI.")


# ✅ **Function: Reset & Create Milvus Collection**
def reset_milvus_collection():
    """Drops existing collection and recreates it."""
    
    if COLLECTION_NAME in utility.list_collections():
        print(f"Dropping existing collection: {COLLECTION_NAME}")
        Collection(COLLECTION_NAME).drop()
    
    print(f"Creating new collection: {COLLECTION_NAME}")

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4096)
    ]
    
    schema = CollectionSchema(fields, description="Document Embeddings")
    collection = Collection(COLLECTION_NAME, schema)
    collection.create_index("embedding", {"metric_type": "COSINE"})  # ✅ Ensure consistency

    print(f"✅ Created new collection: {COLLECTION_NAME} with embedding dim={EMBEDDING_DIM}")
    return collection


# ✅ **Function: Store Embeddings in Milvus**
def store_embeddings(embeddings, texts):
    """Stores embeddings and corresponding text chunks in Milvus."""
    collection = Collection(COLLECTION_NAME)
    if not isinstance(embeddings, list) or not all(isinstance(e, list) for e in embeddings):
        raise ValueError("Embeddings should be a list of lists.")

    if any(len(e) != EMBEDDING_DIM for e in embeddings):
        raise ValueError(f"Each embedding must have {EMBEDDING_DIM} dimensions.")

    if len(embeddings) != len(texts):
        raise ValueError("Number of embeddings must match number of text chunks.")

    collection.insert([embeddings, texts])
    collection.flush()
    print(f"✅ Stored {len(embeddings)} embeddings in '{COLLECTION_NAME}'.")


# ✅ **Function: Search Embeddings in Milvus**
def search_embeddings(query_embedding, top_k=3):
    """Performs similarity search in Milvus."""
    collection = Collection(COLLECTION_NAME)
    collection.load()

    search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}  # ✅ Fixed metric type
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["text"]
    )

    return [hit.entity.get("text") for hit in results[0]]


# 📌 **API Route: Process PDF Files**
@ns_processing.route("/index")
class DocumentIndexer(Resource):
    def post(self):
        """Automatically process all PDFs in the data_input folder."""
        files = [f for f in os.listdir(DATA_INPUT_FOLDER) if f.endswith(".pdf")]
        if not files:
            return {"message": "No PDF files found in data_input folder!"}, 400

        for file_name in files:
            file_path = os.path.join(DATA_INPUT_FOLDER, file_name)
            chunks = extract_text_from_pdf(file_path)
            embeddings = embed_text(chunks)
            store_embeddings(embeddings, chunks)
            shutil.move(file_path, os.path.join(PROCESSED_FOLDER, file_name))

        return {"message": "All documents processed and moved successfully!"}, 200


# 📌 **API Route: Query Documents**
@ns_query.route("/query")
class DocumentQuery(Resource):
    @api.expect(query_model)
    def post(self):
        """Handle user query, perform similarity search, and generate a response from Azure OpenAI."""
        try:
            user_query = request.json.get("query")
            if not user_query:
                return {"message": "Query not provided!"}, 400

            query_embedding = embed_text([user_query])[0]
            top_k_chunks = search_embeddings(query_embedding, top_k=3)

            if not top_k_chunks:
                return {"message": "No relevant information found."}, 200

            augmented_prompt = f"User Query: {user_query}\n\nRelevant Chunks:\n" + "\n".join(top_k_chunks)
            print(augmented_prompt)

            response = client.chat.completions.create(
                model=DEPLOYMENT_CHAT,  # ✅ Fixed model reference
                messages=[{"role": "system", "content": "You just need to repond as mentioned from the releavent chunks, don't add extra information. if you cannot find the answer respond with I don't know"},
                          {"role": "user", "content": augmented_prompt}],
                max_tokens=500,
                temperature= int(os.getenv("Temperature"))
            )

            return {"response": response.choices[0].message.content.strip()}, 200

        except Exception as e:
            print(f"Error occurred: {e}")
            return {"message": f"Error: {str(e)}"}, 500


# Run Flask App
if __name__ == "__main__":
    app.run(debug=True)

#, port=5001