import os
import streamlit as st
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

if not os.path.exists("uploaded"):
    os.mkdir("uploaded")

if not os.path.exists("vectors"):
    os.mkdir("vectors")

class Config:
    FILES_PATH = "uploaded"
    PERSIST_DIRECTORY = "vectors"
    MODEL = "llama3:latest"
    BASE_URL = "http://localhost:11434"
    EMBEDDING = OllamaEmbeddings(model = "llama3:latest")
    EMBEDDING_FUNCTION = OllamaEmbeddings(base_url = "http://localhost:11434", model = "llama3:latest")
    CALLBACK_MANAGER = CallbackManager([StreamingStdOutCallbackHandler()])