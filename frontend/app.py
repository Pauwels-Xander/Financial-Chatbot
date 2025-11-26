import streamlit as st
import requests
import json

# API endpoint (adjust if backend is on a different host/port)
API_URL = "http://localhost:8000/ask"

st.title("Finance QA Chatbot")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history (loop through messages and render bubbles)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input field (persistent at bottom)
if prompt := st.chat_input("Ask a question about financial data (e.g., 'What was the total revenue in 2023?')"):
    # Display user message bubble
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Show loading indicator while calling API
    with st.spinner("Processing your query..."):
        try:
            # Make POST request to /ask
            response = requests.post(API_URL, json={"query": prompt})
            response.raise_for_status()  # Raise error on bad status
            
            data = response.json()
            answer = data.get("answer_text", "No answer received.")
            
            # Display assistant message bubble
            with st.chat_message("assistant"):
                st.markdown(answer)
            
            # Add to history
            st.session_state.messages.append({"role": "assistant", "content": answer})
        
        except requests.exceptions.RequestException as e:
            error_msg = f"Error connecting to backend: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

# clear chat button
if st.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()  # Rerun to refresh UI
