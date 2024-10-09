#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Save your Streamlit code into a file
code = """
import os
import streamlit as st
import pdfplumber
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import openai
from langdetect import detect
from nltk.tokenize import sent_tokenize

# Your OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Streamlit UI for the chatbot
st.set_page_config(page_title="Document-Based Chatbot", layout="wide")

st.title("Document-Based Question Answering Chatbot")
st.markdown("Upload your documents and ask questions!")

# Sidebar for translation option
with st.sidebar:
    st.header("Translation Options")
    translate_option = st.radio("Would you like to translate responses?", ('No', 'Yes'))
    target_language = st.selectbox("Select target language", ['French', 'Spanish', 'German']) if translate_option == 'Yes' else None

# Initialize session state for the chat history
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# File Uploader
uploaded_files = st.file_uploader("Upload PDF documents", type=["pdf"], accept_multiple_files=True)

# Function to extract text from PDFs
@st.cache
def extract_text_from_pdfs(files):
    documents = {}
    for file in files:
        with pdfplumber.open(file) as pdf:
            full_text = ""
            for page in pdf.pages:
                full_text += page.extract_text()
            documents[file.name] = full_text
    return documents

# Function to chunk the text into smaller segments
def chunk_text_for_all_docs(documents, max_tokens=500):
    all_chunks = {}
    for filename, text in documents.items():
        sentences = sent_tokenize(text)
        chunks = []
        chunk = []
        tokens_count = 0
        for sentence in sentences:
            tokens = len(sentence.split())
            if tokens_count + tokens > max_tokens:
                chunks.append(" ".join(chunk))
                chunk = []
                tokens_count = 0
            chunk.append(sentence)
            tokens_count += tokens
        if chunk:
            chunks.append(" ".join(chunk))
        all_chunks[filename] = chunks
    return all_chunks

# Function to generate embeddings
def generate_embeddings_for_all_docs(all_chunks):
    all_embeddings = {}
    embedding_ids = []
    chunk_count = 0
    for filename, chunks in all_chunks.items():
        embeddings = model.encode(chunks, convert_to_tensor=False)
        all_embeddings[filename] = embeddings
        for i, chunk in enumerate(chunks):
            embedding_ids.append(f"{filename}-chunk-{i}")
            chunk_count += 1
    return all_embeddings, embedding_ids

# Function to create FAISS index
def create_faiss_index(embeddings):
    dimension = embeddings[next(iter(embeddings))][0].shape[0]  # Embedding size
    index = faiss.IndexFlatL2(dimension)  # L2 distance index
    all_embedding_list = []
    for embedding_list in embeddings.values():
        all_embedding_list.extend(embedding_list)
    index.add(np.array(all_embedding_list))
    return index

# Function to query FAISS
def query_faiss(query, all_chunks, index, embedding_ids, top_k=3):
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), top_k)
    retrieved_chunks = [all_chunks[embedding_ids[i].split('-chunk-')[0]][int(embedding_ids[i].split('-chunk-')[-1])] for i in I[0]]
    return retrieved_chunks

# Function to generate a response from GPT-3.5
def generate_response_with_context(query, retrieved_chunks):
    prompt = f"User query: {query}\\n\\nRelevant information from documents:\\n{retrieved_chunks}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200
    )
    return response['choices'][0]['message']['content'].strip()

# Function to translate the response
def translate_text(text, target_language):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"Translate this text to {target_language}."},
            {"role": "user", "content": text}
        ],
        max_tokens=100
    )
    return response['choices'][0]['message']['content'].strip()

# Main section for query and responses
if uploaded_files:
    with st.spinner('Processing documents...'):
        documents = extract_text_from_pdfs(uploaded_files)
        all_chunks = chunk_text_for_all_docs(documents)
        all_embeddings, embedding_ids = generate_embeddings_for_all_docs(all_chunks)
        index = create_faiss_index(all_embeddings)
        st.success('Documents processed successfully!')

    # Continuous Chat Interface
    user_query = st.text_input("Enter your query:")

    if st.button("Submit Query") and user_query:
        retrieved_chunks = query_faiss(user_query, all_chunks, index, embedding_ids)
        response = generate_response_with_context(user_query, retrieved_chunks)
        
        # Store the chat history
        st.session_state.chat_history.append(f"You: {user_query}")
        st.session_state.chat_history.append(f"Chatbot: {response}")
        
        # Display chat history
        for msg in st.session_state.chat_history:
            st.write(msg)

        # Handle translation if enabled
        if translate_option == 'Yes' and target_language:
            translated_response = translate_text(response, target_language)
            st.write(f"Translated Response ({target_language}): {translated_response}")

"""
with open("app.py", "w") as file:
    file.write(code)


# In[5]:


get_ipython().system('streamlit run apppp.py')


# In[ ]:




