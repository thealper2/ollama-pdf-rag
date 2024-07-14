import time
import streamlit as st
from config import Config
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain_community.llms.ollama import Ollama
from langchain.chains.retrieval_qa.base import RetrievalQA
from streamlit_js_eval import streamlit_js_eval

from pdf_helper import process_pdf

if "template" not in st.session_state:
    st.session_state.template = """You are a knowledgeable chatbot, here to help with questions of the user. Your tone should be professional and informative.

    Context: {context}
    History: {history}

    User: {question}
    Chatbot:"""

if "prompt" not in st.session_state:
    st.session_state.prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=st.session_state.template,
    )

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True,
        input_key="question"
    )
    
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = Chroma(
        persist_directory=Config.PERSIST_DIRECTORY,
        embedding_function=Config.EMBEDDING_FUNCTION
    )

if "llm" not in st.session_state:
    st.session_state.llm = Ollama(
        base_url=Config.BASE_URL,
        model=Config.MODEL,
        callback_manager=Config.CALLBACK_MANAGER,
        verbose=True,
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

def initialize_qa_chain():
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=st.session_state.llm,
            chain_type="stuff",
            retriever=st.session_state.retriever,
            chain_type_kwargs={
                "prompt": st.session_state.prompt,
                "memory": st.session_state.memory,
                "verbose": True
            },
            verbose=True
        )

def display_chat_history():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["message"])

def handle_user_input(user_input):
    user_message = {
        "role": "user",
        "message": user_input
    }

    st.session_state.messages.append(user_message)
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Assistant is typing..."):
            response = st.session_state.qa_chain(user_input)

        message_placeholder = st.empty()
        full_response = ""
        for chunk in response["result"].split():
            full_response += chunk + " "
            time.sleep(0.05)
            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": ""}]
    streamlit_js_eval(js_expressions="parent.window.location.reload()")

st.sidebar.title("PDF Chatbot")

uploaded_file = st.sidebar.file_uploader("Upload your PDF", type="pdf")

display_chat_history()

st.sidebar.button('Clear', on_click=clear_chat_history)

if uploaded_file is not None:
    with st.status("Analyzing..."):
        process_pdf(uploaded_file)
        initialize_qa_chain()

    if user_input := st.chat_input("You:", key="user_input"):
        handle_user_input(user_input)

else:
    st.write("Please upload a PDF file.")