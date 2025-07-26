import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import asyncio
import sys

# 🛠 Fix event loop issue in Streamlit (especially on Windows / Linux)
if sys.platform.startswith('win') or sys.platform == "linux":
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Define models
EMBEDDING_MODEL = "models/embedding-001"
LLM_MODEL = "gemini-2.5-pro"

def extract_text_from_pdf(pdf):
    reader = PdfReader(pdf)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def split_text(text):
    splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

def create_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY
    )

def create_llm():
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=0.3,
        google_api_key=GOOGLE_API_KEY
    )

def save_vector_store(chunks):
    embeddings = create_embeddings()
    db = FAISS.from_texts(chunks, embeddings)
    db.save_local("faiss_index")

def load_vector_store():
    embeddings = create_embeddings()
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

def main():
    st.set_page_config(page_title="PDF Chatbot 💬")
    st.header("Chat with your PDF 📄")

    uploaded_pdf = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_pdf is not None:
        with st.spinner("Reading PDF..."):
            raw_text = extract_text_from_pdf(uploaded_pdf)
            text_chunks = split_text(raw_text)
            save_vector_store(text_chunks)
            st.success("PDF processed and indexed!")

    query = st.text_input("Ask a question about the PDF")

    if query:
        db = load_vector_store()
        retriever = db.as_retriever()
        llm = create_llm()
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        response = qa.run(query)
        st.write("📌 Answer:", response)

if __name__ == "__main__":
    main()
