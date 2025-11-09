import os
import shutil
import openai
from flask import Flask, request
from flask_restx import Api, Resource, fields
from dotenv import load_dotenv
from config import (
    DATA_INPUT_FOLDER, PROCESSED_FOLDER,
    AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, 
    AZURE_OPENAI_DEPLOYMENT_NAME, AZURE_OPENAI_VERSION, AZURE_OPENAI_API_VERSION
)
from process_pdf import extract_text_from_pdf
from embedding import embed_text
from vector_db import store_embeddings, search_embeddings

# Load environment variables
load_dotenv()

# Initialize Flask App
app = Flask(__name__)

# Initialize Flask-RESTx API with Swagger UI
api = Api(app, 
          title="Document Processing API", 
          description="API for PDF processing, embedding, and Milvus storage", 
          version="1.0")

# Define API namespaces
ns_processing = api.namespace("documents_processing", description="Operations related to document processing")
ns_query = api.namespace("documents_query", description="Operations related to querying documents")

# Configure Azure OpenAI API
openai.api_type = "azure"
openai.api_base = AZURE_OPENAI_ENDPOINT
openai.api_key = AZURE_OPENAI_API_KEY
openai.api_version = AZURE_OPENAI_API_VERSION 

# Define Swagger model for query input
query_model = api.model("QueryModel", {
    "query": fields.String(required=True, description="User query in JSON format")
})

# 1️⃣ API for Document Processing
@ns_processing.route("/index")
class DocumentIndexer(Resource):
    def post(self):
        """Automatically process all PDFs in the data_input folder"""
        files = [f for f in os.listdir(DATA_INPUT_FOLDER) if f.endswith(".pdf")]

        if not files:
            return {"message": "No PDF files found in data_input folder!"}, 400

        for file_name in files:
            file_path = os.path.join(DATA_INPUT_FOLDER, file_name)

            # Extract text from PDF
            chunks = extract_text_from_pdf(file_path)

            # Generate embeddings for extracted text
            embeddings = embed_text(chunks)

            # Store embeddings in Milvus
            store_embeddings(embeddings, chunks)

            # Move processed file to the processed folder
            shutil.move(file_path, os.path.join(PROCESSED_FOLDER, file_name))

        return {"message": "All documents processed and moved successfully!"}, 200

# 2️⃣ API for Querying Documents
@ns_query.route("/query")
class DocumentQuery(Resource):
    @api.expect(query_model)
    def post(self):
        """Handle user query, perform similarity search, and generate a response from Azure OpenAI."""
        try:
            # Step 1: Get the user query
            user_query = request.json.get("query")
            if not user_query:
                return {"message": "Query not provided!"}, 400
            
            print(f"User Query: {user_query}")  # Debugging

            # Step 2: Embed the user's query
            query_embedding = embed_text([user_query])[0]
            
            # Step 3: Search for similar document chunks in Milvus
            top_k_chunks = search_embeddings(query_embedding, top_k=3)
            
            if not top_k_chunks:
                return {"message": "No relevant information found."}, 404
            
            # Step 4: Augment the query with retrieved knowledge
            augmented_prompt = (
                f"You are an intelligent assistant helping a user with their question.\n\n"
                f"User's Query: \"{user_query}\"\n\n"
                f"Relevant Information:\n"
            )

            for idx, chunk in enumerate(top_k_chunks, 1):
                augmented_prompt += f"Chunk {idx}: {chunk}\n"

            augmented_prompt += (
                "\nBased on the provided information, generate a clear, concise, and factual response to the user's query."
                " If the retrieved information is insufficient, indicate that you do not have enough data to answer fully."
            )

            # Step 5: Get response from Azure OpenAI 
            response = openai.ChatCompletion.create(
                engine=AZURE_OPENAI_DEPLOYMENT_NAME,  
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": augmented_prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            # Step 6: Return AI-generated response
            ai_response = response["choices"][0]["message"]["content"].strip()
            print(f"Azure OpenAI Response: {ai_response}")  # Debugging

            return {"response": ai_response}, 200

        except Exception as e:
            print(f"Error occurred: {e}")
            return {"message": f"Error occurred: {str(e)}"}, 500

# Run Flask App
if __name__ == "__main__":
    app.run(debug=True)
