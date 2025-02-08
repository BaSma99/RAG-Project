import os 
import streamlit as st
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain.document_loaders import PyPDFLoader
import google.generativeai as genai
import tempfile
import numpy as np

api = 'Put your API Key Here' 

if api:
    genai.configure(api_key=api)
else:
    st.error("API NOT FOUND")
    
def generate_text(text):
    model=genai.GenerativeModel("gemini-pro")
    response=model.generate_content(text)
    return response.text

st.title("RAG with VectorDatabase and Gemini API")
if "message" not in st.session_state:
    st.session_state.message=[]
    
for message in st.session_state.message:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
upload_file=st.file_uploader("Choose file",type=["pdf"])

if upload_file is not None:
    with tempfile.NamedTemporaryFile(delete=False,suffix=".pdf") as tempfile:
        tempfile.write(upload_file.read())
        temp_file_path=tempfile.name
        
    loader=PyPDFLoader(temp_file_path)
    documents=loader.load()
    
    
    embedding_model=SentenceTransformer("all-MiniLM-L6-v2")
    text=[doc.page_content for doc in documents]
    embeddings=embedding_model.encode(text,show_progress_bar=True)
    embeddings_matrix=np.array(embeddings)

    index=faiss.IndexFlatL2(embeddings_matrix.shape[1])
    index.add(embeddings_matrix)
    st.success("Pdf Processed into the Index")
    user_input=st.chat_input("Please enter Question")
    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        
        st.session_state.message.append({"role":"user","content":user_input})
        
        question_embedding=embedding_model.encode([user_input])
        
        k=1
        distances,indices=index.search(question_embedding,k)
        similar_doc=[documents[i] for i in indices[0]]
        context=""
        for i,doc in enumerate(similar_doc):
            context+=doc.page_content +"\n"
            
        
        prompt=f"You are an assistant who retrieves answers based on the following content:{context}\n\nQuestion:{user_input}"
        
        with st.chat_message("assistant"):
            message_placeholder=st.empty()
            with st.spinner("Generating Answer"):
                response_text=generate_text(prompt)
                message_placeholder.markdown(f"{response_text}")
        st.session_state.message.append({"role":"assistant","content":response_text})
else:
    st.info("Please upload a file to chat with")
