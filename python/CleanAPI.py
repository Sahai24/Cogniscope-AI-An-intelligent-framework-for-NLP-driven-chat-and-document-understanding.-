import os
import spacy
from flask import Flask
from flask_restx import Api, Resource, fields
from werkzeug.middleware.proxy_fix import ProxyFix
from openai import AzureOpenAI
from dotenv import load_dotenv 

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)

api = Api(app, version='1.0', title='Simple API', description='A simple API with Flask-RESTx')
nlp = spacy.load("en_core_web_sm")

ns_query = api.namespace('query', description='Query operations')
ns_summary = api.namespace('summary', description='Summarizer')
ns_sentiment = api.namespace('sentiment', description='Sentiment analysis')
ns_ner = api.namespace('NER', description='Named Entity Recognition')

query_model = api.model('Query', {'query': fields.String(required=True, description='User query')})
summary_model = api.model('Summary', {'text': fields.String(required=True, description='Text to summarize')})
sentiment_model = api.model('Sentiment', {'text': fields.String(required=True, description='Text for sentiment analysis')})
ner_model = api.model('NER', {'text': fields.String(required=True, description='Text for named entity recognition')})

# Function to fetch environment variables with defaults
def get_env_var(var_name, default_value, cast_type):
    value = os.getenv(var_name, default_value)
    return cast_type(value)

def get_openai_response(system_prompt, user_prompt, max_tokens, temperature):
    endpoint = os.getenv("ENDPOINT_URL", "")  
    deployment = os.getenv("DEPLOYMENT_NAME", "gpt-35-turbo")  
    subscription_key = os.getenv("AZURE_OPENAI_API_KEY", "")  
    
    client = AzureOpenAI(
        azure_endpoint=endpoint,  
        api_key=subscription_key,  
        api_version="2024-05-01-preview",
    )
    
    chat_prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    completion = client.chat.completions.create(
        model=deployment,
        messages=chat_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.9,
        frequency_penalty=0,
        presence_penalty=0
    )
    
    return completion.choices[0].message.content

@ns_query.route('/')
class QueryResource(Resource):
    @api.expect(query_model)
    def post(self):
        user_query = api.payload.get("query", "").lower()
        response = get_openai_response(
            "You are an AI assistant that helps people find information in Computer Science.", 
            user_query, 
            max_tokens=get_env_var("QUERY_MAX_TOKENS", 800, int), 
            temperature=get_env_var("QUERY_TEMPERATURE", 0.21, float)
        )
        return {"response": response}

@ns_summary.route('/')
class Summarizer(Resource):
    @api.expect(summary_model)
    def post(self):
        text_to_summarize = api.payload.get("text", "")
        response = get_openai_response(
            "You are an advanced AI summarizer. Generate a concise summary while preserving key points.", 
            f"Summarize the following text: {text_to_summarize}", 
            max_tokens=get_env_var("SUMMARY_MAX_TOKENS", 200, int), 
            temperature=get_env_var("SUMMARY_TEMPERATURE", 0.5, float)
        )
        return {"summary": response}

@ns_sentiment.route('/')
class SentimentResource(Resource):
    @api.expect(sentiment_model)
    def post(self):
        text_to_analyze = api.payload.get("text", "")
        response = get_openai_response(
            "You are an AI that performs sentiment analysis. Identify whether the sentiment is Positive, Negative, or Neutral and explain briefly.", 
            f"Analyze the sentiment of the following text: {text_to_analyze}", 
            max_tokens=get_env_var("SENTIMENT_MAX_TOKENS", 800, int), 
            temperature=get_env_var("SENTIMENT_TEMPERATURE", 0.34, float)
        )
        return {"sentiment": response}

@ns_ner.route('/')
class NERResource(Resource):
    @api.expect(ner_model)
    def post(self):
        text_to_analyze = api.payload.get("text", "")
        response = get_openai_response(
            "You are an AI trained to extract named entities from text. Identify persons, organizations, locations, dates, and other important entities.", 
            f"Extract named entities from the following text:\n\n{text_to_analyze}", 
            max_tokens=get_env_var("NER_MAX_TOKENS", 500, int), 
            temperature=get_env_var("NER_TEMPERATURE", 0.3, float)
        )
        return {"entities": response}

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
