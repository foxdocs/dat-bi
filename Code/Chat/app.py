import streamlit as st
import utils
from utils import *
import os

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

st.title("Talk with Multimodal PDF")

uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf",
    accept_multiple_files=False
)

file_path = os.path.join(data, uploaded_file.name)
print(file_path)


if uploaded_file:
    upload_pdf(uploaded_file)
    text = parse_pdf(file_path, media)
    # text = load_pdf(pdfs_directory + uploaded_file.name)
    chunked = split_pdf_text(text)
    store_pdf_docs(chunked)

    question = st.chat_input()

    if question:
        st.chat_message("user").write(question)
        related_documents = retrieve_docs(question)
        answer = answer_question(question, related_documents)
        st.chat_message("assistant").write(answer)