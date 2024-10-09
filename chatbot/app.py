import os
import pdfplumber
from sentence_transformers import SentenceTransformer
import openai
import numpy as np
from nltk.tokenize import sent_tokenize
import faiss
import streamlit as st
from tqdm.autonotebook import tqdm


# Your OpenAI API Key
openai.api_key = "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"  # Replace with your OpenAI API key

# Initialize the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to extract text from PDFs
def extract_text_from_all_pdfs(folder_path):
    documents = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            file_path = os.path.join(folder_path, filename)
            with pdfplumber.open(file_path) as pdf:
                full_text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:  # Avoid adding None if page has no text
                        full_text += page_text
                documents[filename] = full_text
    return documents

# Function to chunk text
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

# Function to perform FAISS query
def query_faiss(query, all_chunks, index, embedding_ids, top_k=3):
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), top_k)
    retrieved_chunks = [all_chunks[embedding_ids[i].split('-chunk-')[0]][int(embedding_ids[i].split('-chunk-')[-1])] for i in I[0]]
    return retrieved_chunks

# Function to generate response with GPT-3.5
def generate_response_with_context(query, retrieved_chunks):
    prompt = f"User query: {query}\n\nRelevant information from documents:\n{retrieved_chunks}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200
    )
    return response['choices'][0]['message']['content'].strip()

# Function to split text into smaller chunks for translation if necessary
def split_text(text, max_tokens=300):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_tokens = 0
    for sentence in sentences:
        sentence_tokens = len(sentence.split())
        if current_tokens + sentence_tokens > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_tokens = 0
        current_chunk.append(sentence)
        current_tokens += sentence_tokens
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# Modified translation function (translates only the response)
def translate_text(text, target_language):
    text_chunks = split_text(text)
    translated_chunks = []
    for chunk in text_chunks:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"Translate this text to {target_language}."},
                {"role": "user", "content": chunk}
            ],
            max_tokens=1000  # Adjust based on translation needs
        )
        translated_chunks.append(response['choices'][0]['message']['content'].strip())
    return " ".join(translated_chunks)

# Streamlit interface
def main():
    st.title("RAG-based Chatbot with Document Support")
    user_query = st.text_input("Please enter your query:")
    
    folder_path = 'C:/Users/Ishtiyak/Desktop/chatbot/documents'  # Folder with your original documents
    documents = extract_text_from_all_pdfs(folder_path)
    all_chunks = chunk_text_for_all_docs(documents)
    all_embeddings, embedding_ids = generate_embeddings_for_all_docs(all_chunks)
    index = create_faiss_index(all_embeddings)

    if user_query:
        retrieved_chunks = query_faiss(user_query, all_chunks, index, embedding_ids)
        response = generate_response_with_context(user_query, retrieved_chunks)
        st.write(f"Response in English: {response}")
        
        translate_option = st.radio("Do you want to translate the response?", ('No', 'Yes'))
        if translate_option == 'Yes':
            target_language = st.text_input("Enter target language (e.g., 'French', 'Spanish', 'German'):")
            if target_language:
                translated_response = translate_text(response, target_language.lower())
                st.write(f"Translated Response in {target_language}: {translated_response}")

if __name__ == '__main__':
    main()
