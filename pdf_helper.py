import os
import streamlit as st
from config import Config
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma

def copy_file(uploaded_file, file_path):
    file_data =  uploaded_file.read()
    f = open(file_path, "wb")
    f.write(file_data)
    f.close()

    return file_path

def load_pdf(file_path):
    loader = PyPDFLoader(file_path=file_path)
    data = loader.load()
    return data

def split_docs(data, chunk_size=1000, chunk_overlap=30, length_function=len):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=length_function
    )

    documents = text_splitter.split_documents(data)
    return documents

def process_pdf(uploaded_file):
    files_path = Config.FILES_PATH
    file_path = f"{files_path}/{uploaded_file.name}.pdf"
    if not os.path.isfile(file_path):
        file_path = copy_file(uploaded_file, file_path)
        data = load_pdf(file_path)
        documents = split_docs(data)

        st.session_state.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=Config.EMBEDDING
        )

        st.session_state.vectorstore.persist()

    st.session_state.retriever = st.session_state.vectorstore.as_retriever()