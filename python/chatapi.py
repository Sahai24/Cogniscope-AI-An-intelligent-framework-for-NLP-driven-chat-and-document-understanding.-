import os
from flask import Flask
from flask_restx import Api, Resource, fields
from werkzeug.middleware.proxy_fix import ProxyFix
from openai import OpenAI
from dotenv import load_dotenv 
load_dotenv()

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)

api = Api(app, version='1.0', title='Simple API', description='A simple API with Flask-RESTx')

ns = api.namespace('query', description='Query operations')

query_model = api.model('Query', {
    'query': fields.String(required=True, description='User query')
})

hardcoded_responses = {
    "hello": "Hi there! How can I help you?",
    "weather": "The weather is sunny with a chance of rain.",
    "time": "The current time is 2:00 PM.",
    "date": "Today's date is March 13, 2025.",
    "default": "I'm not sure about that, but I can try to help!"
}

@ns.route('/')
class QueryResource(Resource):
    @api.expect(query_model)
    def post(self):
        """Process user query"""
        user_query = api.payload.get("query", "").lower()
        print("api call")
       
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

        completion = client.chat.completions.create(
  model="gpt-4o-mini",
  store=True,
  messages=[
    {"role": "user", "content": "write a haiku about ai"}
  ]
)

        print(completion.choices[0].message);

        response = hardcoded_responses.get(user_query, hardcoded_responses["default"])
        return {"response": response}

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)