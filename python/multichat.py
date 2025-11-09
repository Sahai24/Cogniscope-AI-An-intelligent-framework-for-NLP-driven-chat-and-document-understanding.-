import requests
import streamlit as st

st.title("Multi-Function Chatbot")

# Define API base URL
API_BASE_URL = "http://127.0.0.1:5000"
#APIRAG_BASE_URL = "http://127.0.0.1:5001/"

# Dropdown for selecting API functionality
option = st.selectbox(
    "Choose a function:",
    ("Query", "Summarize", "Sentiment Analysis", "Named Entity Recognition (NER)", "RAG")
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Enter your text here:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        # Determine API endpoint based on selection
        endpoint_map = {
            "Query": "query",
            "Summarize": "summary",
            "Sentiment Analysis": "sentiment",
            "Named Entity Recognition (NER)": "NER",
            #"RAG": "documents_query/query"
        }

        selected_endpoint = endpoint_map[option]
        api_url = f"{API_BASE_URL}/{selected_endpoint}"
        print(selected_endpoint)
        payload = {"query" if selected_endpoint == "query" else "text": prompt}
        print(payload)
        headers = {"Content-Type": "application/json"}

        # Send request to Flask API
        response = requests.post(api_url, json=payload, headers=headers)

        # Handle response
        if response.status_code == 200:
            response_data = response.json()
            response_text = (
                response_data.get("response") or  # For 'query'
                response_data.get("summary") or  # For 'summary'
                response_data.get("sentiment") or  # For 'sentiment'
                response_data.get("entities") or  # For 'NER'
                "No response received."
            )
        else:
            response_text = f"Error: {response.status_code}, {response.text}"
            print(response_text)

        # Display response
        message_placeholder.markdown(response_text)
    
    # Save response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_text})
