# user_chat.py

import requests
import streamlit as st
import uuid

API_URL = "http://middle_layer:8000"

def send_to_api(endpoint, data=None):
    try:
        response = requests.post(f"{API_URL}/{endpoint}", json=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")
        return None

st.set_page_config(page_title="LCNC Guide Chat", page_icon=":robot:")
st.title("LCNC Guide Chat")

# Initialize session states
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())  # Automatically generate session ID for the user
    st.write(f"Generated session ID: {st.session_state.session_id}")

if "current_step" not in st.session_state:
    st.session_state.current_step = 1

if "business_category" not in st.session_state:
    st.session_state.business_category = None

if "primary_product_service" not in st.session_state:
    st.session_state.primary_product_service = None

if "target_customer" not in st.session_state:
    st.session_state.target_customer = None

if "user_roles" not in st.session_state:
    st.session_state.user_roles = None

if "template_selected" not in st.session_state:
    st.session_state.template_selected = None

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome! Let's get started with a few questions to better understand your needs. What is your company name?"}
    ]

def handle_user_input(prompt):
    st.session_state.messages.append({"role": "user", "content": prompt})

    if st.session_state.current_step == 1:
        st.session_state.business_category = prompt
        response = send_to_api('ask', data={"session_id": st.session_state.session_id, "query": "Great! What is the primary product or service your business offers?"})
        st.session_state.messages.append({"role": "assistant", "content": response["response"] if response else "I didn't get that. Can you please provide more details?"})
        st.session_state.current_step += 1
        st.session_state.place_holder_prompt = 'Online food delivery from restaurants to customers houses'
        st.write(f"Current step: {st.session_state.current_step}, Business Category: {st.session_state.business_category}")

    elif st.session_state.current_step == 2:
        st.session_state.primary_product_service = prompt
        response = send_to_api('ask', data={"session_id": st.session_state.session_id, "query": "Who is your target customer: individuals, other businesses, or both?"})
        st.session_state.messages.append({"role": "assistant", "content": response["response"] if response else "I didn't get that. Can you please provide more details?"})
        st.session_state.current_step += 1
        st.session_state.place_holder_prompt = 'Delivery directly to customer from restaurant'
        st.write(f"Current step: {st.session_state.current_step}, Primary Product/Service: {st.session_state.primary_product_service}")

    elif st.session_state.current_step == 3:
        st.session_state.target_customer = prompt
        response = send_to_api('ask', data={"session_id": st.session_state.session_id, "query": "Could you please tell me more about the user roles in your app?"})
        st.session_state.messages.append({"role": "assistant", "content": response["response"] if response else "I didn't get that. Can you please provide more details?"})
        st.session_state.current_step += 1
        st.session_state.place_holder_prompt = 'restaurant owners, customers, admin'
        st.write(f"Current step: {st.session_state.current_step}, Target Customer: {st.session_state.target_customer}")

    elif st.session_state.current_step == 4:
        st.session_state.user_roles = prompt
        response = send_to_api('ask', data={"session_id": st.session_state.session_id, "query": "Based on your business needs, which type of app template do you feel would best suit your vision?"})
        st.session_state.messages.append({"role": "assistant", "content": response["response"] if response else "I didn't get that. Can you please provide more details?"})
        st.session_state.current_step += 1
        st.session_state.place_holder_prompt = 'A'
        st.write(f"Current step: {st.session_state.current_step}, User Roles: {st.session_state.user_roles}")

    elif st.session_state.current_step == 5:
        st.session_state.template_selected = prompt
        st.session_state.messages.append({"role": "assistant", "content": "Thank you! Your information has been saved. You can now customize your pages in the LCNC platform."})
        st.session_state.current_step += 1
        st.session_state.place_holder_prompt = None  # End of structured flow
        st.write(f"Current step: {st.session_state.current_step}, Template Selected: {st.session_state.template_selected}")

    else:
        # Proceed with normal Q&A after the structured flow
        response = send_to_api('ask', data={"session_id": st.session_state.session_id, "query": prompt})
        if response and "response" in response:
            st.session_state.messages.append({"role": "assistant", "content": response["response"]})
        else:
            st.session_state.messages.append({"role": "assistant", "content": "I didn't get that. Can you please provide more details?"})

# Chat input handling
prompt = st.chat_input(st.session_state.get('place_holder_prompt', "Your response here"))
if prompt:
    handle_user_input(prompt)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
