import os
from openai import AzureOpenAI
from config import AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT_NAME

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version="2023-03-15-preview"
)

def embed_text(chunks):
    """Generate embeddings using Azure OpenAI."""
    response = client.embeddings.create(input=chunks, model=AZURE_OPENAI_DEPLOYMENT_NAME)

    if hasattr(response, "data") and isinstance(response.data, list):
        return [item.embedding for item in response.data]  # Extract embeddings
    
    raise ValueError("Azure OpenAI response structure is invalid!")
