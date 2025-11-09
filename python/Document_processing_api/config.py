import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Azure OpenAI Configuration
USE_AZURE = True  # Set to False to use SentenceTransformers
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("ENDPOINT_URL")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME","text-embedding-ada-002")
AZURE_OPENAI_VERSION=os.getenv("AZURE_OPENAI_VERSION","gpt-35-turbo")
AZURE_OPENAI_API_VERSION=("AZURE_OPENAI_API_VERSION","2023-03-15-preview")


# Milvus Configuration
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "document_embeddings"

# Folder Paths
DATA_INPUT_FOLDER = "data_input"
PROCESSED_FOLDER = "processed"

# Ensure folders exist
os.makedirs(DATA_INPUT_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
