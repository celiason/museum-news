import streamlit as st
import time
import pandas as pd
import numpy as np
import torch
from streamlit_extras.app_logo import add_logo

# Setup credentials
HF_KEY = st.secrets['HF_KEY']

# Initiate
st.set_page_config(
    page_title="Voices from the Field")

# Load custom functions
from src.llm import augment_prompt, llm

# Add a sidebar
# with st.sidebar:
#     add_radio = st.radio(
#         "Choose a model",
#         ("RAG", "Base LLM"))

# Add logo
st.logo("./assets/fm_logo.png", size="large")

# Formatting?
st.markdown("""
<style>
    .st-emotion-cache-1rtdyuf {
        color: #FFFFFF;
    }

    .st-emotion-cache-1egp75f {
        color: #FFFFFF;
    }

    .st-emotion-cache-1rtdyuf {
        color: #FFFFFF;
    }
</style>
""", unsafe_allow_html=True)

st.title("Voices from the Field")

st.markdown("""
Welcome to Voices from the Field! This website contains an AI-enabled 
chatbot that will respond to any question you may have about the Field
Museum based on our historical documents and news letters from the early 
20th century into the late 90s. Feel free to type your question in the box
below!

""")

# Add image
st.image("assets/field-museum-bw.png")


# Initialize chat history
if "messages_museum" not in st.session_state:
    st.session_state.messages_museum = []

# Display chat messages from history on app rerun (happens for each query)
for message in st.session_state.messages_museum:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        st.caption(message["documents"])

# Ask for user input
# prompt="Tell me about gorillas"
if prompt := st.chat_input("Ask me a question"):
    
    # Add user input to chat history
    st.session_state.messages_museum.append({"role":"user", "content": prompt, "documents": ""})

    # Display user message in a nicely formatted way
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display AI assistant response (assistant gives us a logo)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        with st.spinner("Thinking..."):
            rag = augment_prompt(prompt, top_k=7)
            context = rag[0]
            documents = rag[1]

            # Feed prompt and context into the LLM model
            llm_output = llm(prompt=prompt, context=context, model='llama3', api_key=HF_KEY)

        # if llm.stop == 1:
        #     st.stop()

        # Simulate stream of response with milliseconds delay
        for chunk in llm_output.split():
            full_response += chunk + " "
            time.sleep(0.05)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
        
        message_placeholder.markdown(full_response)
        st.caption(documents)

    # Add assistant response to chat history
    st.session_state.messages_museum.append(
        {"role": "assistant", "content": full_response, "documents": documents})

