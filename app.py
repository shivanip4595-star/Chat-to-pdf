import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)

# Gemini Pro LLM (v1beta-supported)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0.3,
    google_api_key=GOOGLE_API_KEY
)

# Prompt template
custom_prompt_template = """You are a helpful assistant. Use the following context to answer the question.
If you don't know the answer, say "I don't know."

Context:
{context}

Question:
{question}

Answer:"""

PROMPT = PromptTemplate(
    template=custom_prompt_template,
    input_variables=["context", "question"]
)

# PDF processing
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text()
    return text

# Split text into chunks
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    return splitter.split_text(text)

# Save vector store
def save_vector_store(chunks):
    db = FAISS.from_texts(chunks, embedding=embeddings)
    db.save_local("faiss_index")

# Load vector store
def load_vector_store():
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# QA Chain
def get_qa_chain():
    return load_qa_chain(llm, chain_type="stuff", prompt=PROMPT)

# Handle user input
def handle_user_input(query):
    db = load_vector_store()
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    result = qa({"query": query})
    st.write("### Answer:")
    st.write(result["result"])

# Streamlit UI
def main():
    st.set_page_config("Chat with PDF - Gemini")
    st.title("📄 Chat with PDF using Gemini Pro")

    user_question = st.text_input("Ask a question based on your PDF:")
    if user_question:
        handle_user_input(user_question)

    with st.sidebar:
        st.header("Upload PDF files")
        pdf_docs = st.file_uploader("Choose PDF(s)", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                chunks = get_text_chunks(raw_text)
                save_vector_store(chunks)
                st.success("✅ PDFs processed and vector store saved!")

if __name__ == "__main__":
    main()
