import requests
import streamlit as st

st.title("ChatGPT-like Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Make a POST request with JSON payload
        api_url = "http://127.0.0.1:5000/query/"
        payload = {"query": prompt}  # Include the user input in request body
        headers = {"Content-Type": "application/json"}

        response = requests.post(api_url, json=payload, headers=headers)

        if response.status_code == 200:
            response_text = response.json().get("response", "No response received.")
        else:
            response_text = f"Error: {response.status_code}, {response.text}"

        message_placeholder.markdown(response_text)
    
    # Save response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_text})
