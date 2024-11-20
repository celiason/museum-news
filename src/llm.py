import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
import streamlit as st

DATA_URL = "celiason1/museum"
# LLM_MODEL = "tiiuae/falcon-7b-instruct"
LLM_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

# TODO make this model selectable below

@staticmethod
@st.cache_data(show_spinner=False)
def load_data(data_source):
    """Loads and returns a dataset from a specified data URL using
    Hugging Face's datasets library.

    Returns:
    - dataset: a Pandas DataFrame containing the loaded dataset
    """
    dataset = load_dataset(
        DATA_URL, download_mode='force_redownload',
        verification_mode='no_checks')['train'].to_pandas()
    return dataset

@staticmethod
@st.cache_data()
def get_embeddings(data):
    not_embedding_columns = ['index', 'title', 'text_chunk', 'new_column']
    # Convert embeddings to tensor
    res = data.drop(columns=not_embedding_columns)
    res = torch.from_numpy(res.to_numpy()).to(torch.float)
    return res

# Load the data
data = load_data(DATA_URL)
data = data.loc[data['0'].notna()]

# Get just the embeddings
embeddings = get_embeddings(data)

# Generating augmented prompts
def augment_prompt(prompt, top_k=10):
    """
    prompt = 'tell me about okapis at the museum'

    """
    
    from sentence_transformers import SentenceTransformer    
    from sentence_transformers.util import semantic_search
    
    ST = SentenceTransformer('all-MiniLM-L6-v2')
    # ST = SentenceTransformer('BAAI/bge-small-en-v1.5')
    # ST = SentenceTransformer('Alibaba-NLP/gte-base-en-v1.5', trust_remote_code=True)
    query_embeddings = ST.encode(prompt)

    # Search the query against the augmented database
    hits = semantic_search(
        query_embeddings,
        embeddings,
        top_k=top_k)

    # Pull out text of interest based on hits
    selected_rows = [hits[0][i]['corpus_id'] for i in range(len(hits[0]))]
    
    # Set text as the context for the LLM
    context = data.loc[selected_rows, 'new_column'].values.tolist()
    context = "\n\n".join([x for x in context])

    # Find the PDF docs that correspond to the retrieved information
    documents = data.loc[selected_rows,'title'].values.tolist()
    docs = list(set([x.replace(".pdf", "") for x in documents]))
    docs = "\n".join([f"• {item}" for item in docs])
    
    return context, docs

# Show the LLM name in the sidebar
with st.sidebar:
    st.write("LLM model: `", LLM_MODEL, "`")

# Setup the LLM chatbot
def llm(prompt, context, model, api_key, top_k=10):
    """
    Sample prompt: "Tell me about gorillas at the field museum."
    outputs chatbot response
    """

    # RAG step
    # prompt_aug = augment_prompt(prompt, top_k = top_k)
    # context = prompt_aug[0]

    augmented_prompt = f"""

    Using the following context, answer the question. If you don't know the 
    answer, say that you don't know, don't try to make up an answer. Imagine 
    that you work in a museum  and you are cordially answering visitors' questions.
    Respond by saying hello on behalf of the Field Museum and thanking them for asking.
    Answer the question in English.

    Context: {context}

    Question: {prompt}

    """

    messages = [{"role": "user", "content": augmented_prompt}]
    client = InferenceClient(LLM_MODEL, api_key=api_key)
    result = client.chat_completion(messages, max_tokens=500, stream=False)
    
    output = result.choices[0].message.content
    
    return output
