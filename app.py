import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini embeddings
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )

# Use Gemini Pro (correct model name!)
def get_llm():
    return ChatGoogleGenerativeAI(
        model="models/gemini-pro",  # ✅ This is the correct model name
        google_api_key=GOOGLE_API_KEY
    )

# Extract text from PDF
def extract_text_from_pdf(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

# Split text into chunks
def split_text(text):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return splitter.split_text(text)

# Create vector store
def create_vector_store(text_chunks):
    embeddings = get_embeddings()
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

# Ask question
def get_response(db, query):
    llm = get_llm()
    retriever = db.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False
    )
    return qa_chain.run(query)

# Streamlit app
def main():
    st.set_page_config(page_title="Chat with PDF - Gemini", layout="wide")
    st.title("📄 Chat with your PDF using Gemini")
    st.markdown("Upload one or more PDF files and ask questions about them!")

    with st.sidebar:
        st.header("📎 Upload PDFs")
        pdf_docs = st.file_uploader("Upload your PDF files", accept_multiple_files=True)
        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Reading and indexing..."):
                    raw_text = extract_text_from_pdf(pdf_docs)
                    chunks = split_text(raw_text)
                    vector_store = create_vector_store(chunks)
                    st.session_state.vector_store = vector_store
                st.success("✅ PDF processed successfully!")

    query = st.text_input("💬 Ask a question about your PDF")
    if query:
        if "vector_store" not in st.session_state:
            st.error("Please upload and process a PDF first.")
        else:
            with st.spinner("Thinking..."):
                response = get_response(st.session_state.vector_store, query)
                st.write("🧠 Answer:")
                st.markdown(response)

if __name__ == "__main__":
    main()
