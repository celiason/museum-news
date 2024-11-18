import streamlit as st
import time
import pandas as pd
import numpy as np
import torch
from streamlit_extras.app_logo import add_logo

# Initiate
st.set_page_config(
    page_title="Ask the Field Museum", page_icon=":bird:")

# Load custom functions
from src.llm import augment_prompt, llm

# Add a sidebar
with st.sidebar:
    add_radio = st.radio(
        "Choose a model",
        ("RAG", "Base LLM"))
    

# Add logo
st.logo("./unnamed.png", size="large")

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

st.title("History of the Field Museum :bird:")

# Add image
st.image("field-museum-chicago-illinois-nby-414913-4e75aa-640.jpg")


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
            llm_output = llm(prompt=prompt, context=context, model='llama3')

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
