import os 
import spacy
from flask import Flask
from flask_restx import Api, Resource, fields
from werkzeug.middleware.proxy_fix import ProxyFix
from openai import AzureOpenAI
from dotenv import load_dotenv 
load_dotenv()

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)

api = Api(app, version='1.0', title='Simple API', description='A simple API with Flask-RESTx')
nlp = spacy.load("en_core_web_sm")

ns_query = api.namespace('query', description='Query operations')
ns_summary = api.namespace('summary', description='Summarizer')
ns_sentiment = api.namespace('sentiment', description='Sentiment analysis')
ns_ner = api.namespace('NER', description='Named Entity Recognition')

query_model = api.model('Query', {
    'query': fields.String(required=True, description='User query')})

summary_model = api.model('Summary', {
    'text': fields.String(required=True, description='Text to summarize')})

sentiment_model = api.model('Sentiment', {
    'text': fields.String(required=True, description='Text for sentiment analysis')})

ner_model = api.model('NER', {
    'text': fields.String(required=True, description='Text for named entity recognition')})

'''
hardcoded_responses = {
    "hello": "Hi there! How can I help you?",
    "weather": "The weather is sunny with a chance of rain.",
    "time": "The current time is 2:00 PM.",
    "date": "Today's date is March 13, 2025.",
    "default": "I'm not sure about that, but I can try to help!"
}
'''
@ns_query.route('/')
class QueryResource(Resource):
    @api.expect(query_model)
    def post(self):
        """Process user query"""
        user_query = api.payload.get("query", "").lower()
        print("api call")
       
        endpoint = os.getenv("ENDPOINT_URL", "")  
        deployment = os.getenv("DEPLOYMENT_NAME", "gpt-35-turbo")  
        subscription_key = os.getenv("AZURE_OPENAI_API_KEY", "")  
 
        # Initialize Azure OpenAI Service client with key-based authentication    
        client = AzureOpenAI(  
        azure_endpoint=endpoint,  
        api_key=subscription_key,  
        api_version="2024-05-01-preview",
)

        #Prepare the chat prompt 
        chat_prompt = [
        {
            "role": "system",
            "content": "You are an AI assistant that helps people find information in Computer Science."
        },
        {
            "role": "user",
            "content": f"{user_query}"
        }
] 
        # Include speech result if speech is enabled  
        messages = chat_prompt  
        # Generate the completion  
        completion = client.chat.completions.create(  
            model=deployment,
            messages=messages,
            max_tokens=800,  
            temperature=0.21,  
            top_p=0.95,  
            frequency_penalty=0,  
            presence_penalty=0,
            stop=None,  
            stream=False
)
 
        print(completion.to_json())

        print(completion.choices[0].message.content);

        response = completion.choices[0].message.content
        return {"response": response}
@ns_summary.route('/')
class Summarizer(Resource):
    @api.expect(summary_model)
    def post(self):
        text_to_summarize = api.payload.get("text", "")
        
        endpoint = os.getenv("ENDPOINT_URL", "")  
        deployment = os.getenv("DEPLOYMENT_NAME", "gpt-35-turbo")  
        subscription_key = os.getenv("AZURE_OPENAI_API_KEY", "")  
    
        client = AzureOpenAI(  
            azure_endpoint=endpoint,  
            api_key=subscription_key,  
            api_version="2024-05-01-preview",
        )

        chat_prompt = [
            {"role": "system", 
             "content": "You are an advanced AI summarizer. Your task is to analyze the given text and generate a concise summary while preserving key points, important details, and maintaining the original context. Ensure that the summary is clear, coherent, and does not omit critical information."},
            
            {"role": "user", 
             "content": f"Summarize the following text: {text_to_summarize}"}
        ] 
        
        completion = client.chat.completions.create(  
            model=deployment,
            messages=chat_prompt,
            max_tokens=200,  
            temperature=0.5,  
            top_p=0.9,  
            frequency_penalty=0,  
            presence_penalty=0,
            stop=None,  
            stream=False
        )
        
        print(completion.to_json())

        print(completion.choices[0].message.content);
        
        response = completion.choices[0].message.content
        return {"summary": response}
@ns_sentiment.route('/')
class sentiment(Resource):
    @api.expect(sentiment_model)
    def post(self):
        text_to_analyze = api.payload.get("text", "")
        
        endpoint = os.getenv("ENDPOINT_URL", "")  
        deployment = os.getenv("DEPLOYMENT_NAME", "gpt-35-turbo")  
        subscription_key = os.getenv("AZURE_OPENAI_API_KEY", "")  
    
        client = AzureOpenAI(  
            azure_endpoint=endpoint,  
            api_key=subscription_key,  
            api_version="2024-05-01-preview",
        )

        chat_prompt = [
            {"role": "system", "content": "You are an AI that performs sentiment analysis. Identify whether the sentiment is Positive, Negative, or Neutral and explain briefly."},
            {"role": "user", "content": f"Analyze the sentiment of the following text: {text_to_analyze}"}
        ] 
        
        completion = client.chat.completions.create(  
            model=deployment,
            messages=chat_prompt,
            max_tokens=800,  
            temperature=0.34,  
            top_p=0.9,  
            frequency_penalty=0,  
            presence_penalty=0,
            stop=None,  
            stream=False
        )
        
        response = completion.choices[0].message.content
        return {"sentiment": response}
@ns_ner.route('/')
class NERResource(Resource):
    @api.expect(ner_model)
    def post(self):
        """Perform Named Entity Recognition using OpenAI"""
        text_to_analyze = api.payload.get("text", "")
        
        endpoint = os.getenv("ENDPOINT_URL", "")  
        deployment = os.getenv("DEPLOYMENT_NAME", "gpt-35-turbo")  
        subscription_key = os.getenv("AZURE_OPENAI_API_KEY", "")  

        client = AzureOpenAI(  
            azure_endpoint=endpoint,  
            api_key=subscription_key,  
            api_version="2024-05-01-preview",
        )

        chat_prompt = [
            {"role": "system", "content": "You are an AI trained to extract named entities from text. Identify persons, organizations, locations, dates, and other important entities."},
            {"role": "user", "content": f"Extract named entities from the following text:\n\n{text_to_analyze}"}
        ] 
        
        completion = client.chat.completions.create(  
            model=deployment,
            messages=chat_prompt,
            max_tokens=500,  
            temperature=0.3,  
            top_p=0.9,  
            frequency_penalty=0,  
            presence_penalty=0
        )

        response = completion.choices[0].message.content
        return {"entities": response}
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)

