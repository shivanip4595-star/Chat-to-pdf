import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Setup Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

# Read PDF and split into chunks
def extract_text_chunks(pdf_file):
    reader = PdfReader(pdf_file)
    raw_text = ""
    for page in reader.pages:
        raw_text += page.extract_text() or ""
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(raw_text)
    return chunks

# Save FAISS vector store
def save_vector_store(text_chunks):
    vectordb = FAISS.from_texts(text_chunks, embedding=embeddings)
    vectordb.save_local("faiss_index")

# Load FAISS index
def load_vector_store():
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Streamlit UI
def main():
    st.set_page_config(page_title="Chat with your PDF (Gemini)", layout="wide")
    st.title("📄 Chat with your PDF using Gemini")
    
    if "db_loaded" not in st.session_state:
        st.session_state.db_loaded = False

    pdf_file = st.file_uploader("Upload your PDF", type=["pdf"])

    if pdf_file is not None and st.button("Process PDF"):
        with st.spinner("Processing PDF..."):
            chunks = extract_text_chunks(pdf_file)
            save_vector_store(chunks)
            st.session_state.db_loaded = True
            st.success("PDF processed and vector index created!")

    if st.session_state.db_loaded:
        query = st.text_input("Ask a question from your PDF:")
        if query:
            db = load_vector_store()
            docs = db.similarity_search(query)
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=query)
            st.markdown("### 🤖 Answer:")
            st.write(response)

if __name__ == "__main__":
    main()
