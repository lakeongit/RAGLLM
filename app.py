# Install Packages
pip install openai langchain streamlit pandas pymupdf python-docx

# Build the RAG Pipeline with Langchain
import pandas as pd
import fitz  # PyMuPDF
import docx
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter

def load_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def load_docx(file_path):
    doc = docx.Document(file_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def load_csv(file_path):
    df = pd.read_csv(file_path)
    return df.to_string()

def load_xlsx(file_path):
    df = pd.read_excel(file_path)
    return df.to_string()

def ingest_document(file_path):
    if file_path.endswith('.pdf'):
        return load_pdf(file_path)
    elif file_path.endswith('.docx'):
        return load_docx(file_path)
    elif file_path.endswith('.csv'):
        return load_csv(file_path)
    elif file_path.endswith('.xlsx'):
        return load_xlsx(file_path)
    else:
        raise ValueError("Unsupported file format")

def create_rag_pipeline(documents):
    texts = [ingest_document(doc) for doc in documents]
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(texts)

    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_documents(docs, embeddings)
    return docsearch

# Build the Streamlit UI

import streamlit as st
from langchain.chains import RAGChain
from openai import OpenAI

st.set_page_config(page_title="InfoSec Chatbot", layout="wide")

st.title("Information Security Chatbot")

uploaded_files = st.file_uploader("Upload your documents", accept_multiple_files=True)
query = st.text_input("Ask a question about Information Security")

if st.button("Submit"):
    if uploaded_files:
        documents = [file.name for file in uploaded_files]
        docsearch = create_rag_pipeline(documents)
        chat_chain = RAGChain(docsearch, OpenAI(temperature=0.7))

        response = chat_chain.run(query)
        st.write(response)
    else:
        st.error("Please upload documents to proceed.")

