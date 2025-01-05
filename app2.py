pip install streamlit langchain openai faiss-cpu python-docx pypdf pandas openpyxl

import streamlit as st
from langchain.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    CSVLoader,
    UnstructuredExcelLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from typing import List, Dict
import os
from pathlib import Path
import tempfile

# Configuration and Setup
class Config:
    OPENAI_API_KEY = "your-openai-api-key"  # Store this securely
    ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.csv', '.xlsx'}
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        self.embeddings = OpenAIEmbeddings(openai_api_key=Config.OPENAI_API_KEY)

    def load_document(self, file_path: str):
        """Load document based on file extension"""
        ext = Path(file_path).suffix.lower()
        if ext == '.pdf':
            loader = PyPDFLoader(file_path)
        elif ext == '.docx':
            loader = Docx2txtLoader(file_path)
        elif ext == '.csv':
            loader = CSVLoader(file_path)
        elif ext == '.xlsx':
            loader = UnstructuredExcelLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        
        return loader.load()

    def process_documents(self, files: List[str]) -> FAISS:
        """Process multiple documents and create a vector store"""
        documents = []
        for file_path in files:
            documents.extend(self.load_document(file_path))

        texts = self.text_splitter.split_documents(documents)
        vector_store = FAISS.from_documents(texts, self.embeddings)
        return vector_store

class InfoSecChatbot:
    def __init__(self, vector_store: FAISS):
        self.llm = ChatOpenAI(
            temperature=0.2,
            model_name="gpt-4",
            openai_api_key=Config.OPENAI_API_KEY
        )
        self.memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True
        )
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=vector_store.as_retriever(),
            memory=self.memory,
            verbose=True
        )

    def get_response(self, query: str) -> str:
        """Get response from the chatbot"""
        response = self.chain({"question": query})
        return response['answer']

class StreamlitUI:
    def __init__(self):
        st.set_page_config(page_title="InfoSec Chatbot", layout="wide")
        self.initialize_session_state()

    @staticmethod
    def initialize_session_state():
        """Initialize session state variables"""
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'chatbot' not in st.session_state:
            st.session_state.chatbot = None

    def render_ui(self):
        """Render the Streamlit UI"""
        st.title("Information Security Chatbot")
        st.sidebar.title("Document Upload")

        # File uploader in sidebar
        uploaded_files = st.sidebar.file_uploader(
            "Upload your documents",
            accept_multiple_files=True,
            type=['pdf', 'docx', 'csv', 'xlsx']
        )

        if uploaded_files and not st.session_state.chatbot:
            with st.spinner("Processing documents..."):
                # Save uploaded files temporarily
                temp_dir = tempfile.mkdtemp()
                temp_files = []
                for file in uploaded_files:
                    temp_path = os.path.join(temp_dir, file.name)
                    with open(temp_path, 'wb') as f:
                        f.write(file.getvalue())
                    temp_files.append(temp_path)

                # Process documents
                processor = DocumentProcessor()
                vector_store = processor.process_documents(temp_files)
                st.session_state.chatbot = InfoSecChatbot(vector_store)
                st.sidebar.success("Documents processed successfully!")

        # Chat interface
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask me about Information Security"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                if st.session_state.chatbot:
                    response = st.session_state.chatbot.get_response(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    st.warning("Please upload documents to begin the conversation.")

def main():
    ui = StreamlitUI()
    ui.render_ui()

if __name__ == "__main__":
    main()
