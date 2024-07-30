# app.py

import os
import tempfile
import requests
import streamlit as st
from streamlit_chat import message

# Define the base URL for the Ollama model
API_URL = "http://middle_layer:8000"

# Call chat_pdf API
def send_to_api(endpoint, data=None, files=None):
    try:
        response = requests.post(f"{API_URL}/{endpoint}", json=data, files=files)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")
        return None

st.set_page_config(page_title="Chat with Ollama & PDF", page_icon=":robot:")
st.title("Chat with Ollama & PDF")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question or upload a PDF document!"}
    ]

if "assistant" not in st.session_state:
    st.session_state.assistant = None

# File uploader
def read_and_save_file():
    if st.session_state["file_uploader"]:
        st.session_state["assistant"] = None
        st.session_state["messages"] = []
        st.session_state["user_input"] = ""

        for file in st.session_state["file_uploader"]:
            with tempfile.NamedTemporaryFile(delete=False) as tf:
                tf.write(file.getbuffer())
                file_path = tf.name

            with st.spinner(f"Ingesting {file.name}"):
                response = send_to_api('ingest', files={"file": open(file_path, 'rb')})
                os.remove(file_path)

                if response and "message" in response:
                    st.write(response["message"])
                else:
                    st.write("Failed to ingest PDF")

st.file_uploader(
    "Upload a PDF document",
    type=["pdf"],
    key="file_uploader",
    on_change=read_and_save_file,
    label_visibility="collapsed",
    accept_multiple_files=True,
)

# Display chat messages
if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Generate response
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = send_to_api('ask', data={"query": prompt})
            response_text = response["response"] if response and "response" in response else "Failed to get response"
            st.write(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
