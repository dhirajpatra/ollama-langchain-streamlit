from langchain_community.llms import Ollama
import streamlit as st
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


# Define the base URL for the model
llm = Ollama(model="phi3:latest", base_url="http://ollama:11434", verbose=True)

# Define a function to send the prompt to the model and return the response
def sendPrompt(prompt):
    global llm
    response = llm.invoke(prompt)
    return response

# Define the Streamlit app
st.set_page_config(page_title="Chat with Ollama", page_icon=":robot:")
st.title("Chat with Ollama")
if "messages" not in st.session_state.keys(): 
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question !"}
    ]

# Display the chat messages
if prompt := st.chat_input("Your question"): 
    st.session_state.messages.append({"role": "user", "content": prompt})

# Display the chat messages
for message in st.session_state.messages: 
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
# If the last message is not from the assistant, generate a new response
print(st.session_state.messages)
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = sendPrompt(prompt)
            print(response)
            st.write(response)
            message = {"role": "assistant", "content": response}
            st.session_state.messages.append(message)
