import streamlit as st
import utils
from utils import *
import os, time

# A template for the dialoque
template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""

# directories
data = 'data/'
media = 'media/'

uploaded_file = ''

# Page configuration
st.set_page_config(
    page_title="RAG Chat",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

def available_types():
    return ["pdf-based RAG", "web-based RAG"]

def available_models():
    return ["llama3.2:3b", "gemma3:12b", "deepseek-r1:8b", ]


def main():
    st.title("RAG Chat")
    st.markdown("<p style='text-align: left; color: #666;'>Powered by Ollama models, langchain, and unstructured</p>", unsafe_allow_html=True)

    # Sidebar controls
    with st.sidebar:
        st.header("Config")
        
        embed_model = st.selectbox(
            "Select base embedding model",
            available_models(),
            index=0,
            help="Choose embedding model"
        )

        lang_model = st.selectbox(
            "Select base language model",
            available_models(),
            index=1,
            help="Choose multimodal language model"
        )
    
        embeddings = OllamaEmbeddings(model=embed_model)
        vecDB = InMemoryVectorStore(embeddings)
        llm = OllamaLLM(model = lang_model)

    # Main content area with two tabs
    tab1, tab2 = st.tabs(["PDF RAG", "WEB RAG"])
    
    
    with tab1:
        tab1.empty()
        st.header("Talk With Multimodal PDF")
        
        uploaded_file = st.file_uploader(
            "Drop your file here",
            type="pdf",
            accept_multiple_files=False,
            help="The pdf file can contain text, images, tables."
        )  
    
        if uploaded_file:
            file_path = os.path.join(data, uploaded_file.name)
            upload_pdf(uploaded_file)

            with st.status("Working", expanded=True) as status:
                st.write("Reading and parsing the source documents ...")
                text = parse_pdf(file_path, media)
                st.write("Extracting and chunking the document ...")
                chunked = split_pdf_text(text)
                st.write("Embedding and storing the text into a vector database ...")
                store_pdf_docs(chunked, vecDB)
                status.update(label="Completed", state="complete", expanded=False)
     
            question = st.chat_input("Type in here")        
            if question:
                with st.spinner("Generating the answer ..."):
                    st.chat_message("user").write(question)
                    related_documents = retrieve_docs(question, vecDB)
                    answer = answer_question(question, related_documents, llm)
                    st.chat_message("assistant").write(answer)
        else:
            pass     
            
    with tab2:
        tab2.empty()
        st.header("Talk With Web")
        url = st.text_input("Enter URL:")

        with st.status("Working", expanded=True) as status:
                    st.write("Reading and parsing the source documents ...")
                    text = load_web_page(url)
                    st.write("Extracting and chunking the document ...")
                    chunked = split_web_text(text)
                    st.write("Embedding and storing the text into a vector database ...")
                    store_web_docs(chunked, vecDB)
                    status.update(label="Completed", state="complete", expanded=False)

        question = st.chat_input("Enter your message here")
        if question:
                with st.spinner("Generating the answer ..."):
                    st.chat_message("user").write(question)
                    retrieved = retrieve_docs(question, vecDB)
                    answer = answer_question(question, retrieved, llm)
                    st.chat_message("assistant").write(answer)

if __name__ == "__main__":
    main()