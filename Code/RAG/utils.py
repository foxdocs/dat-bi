# Utility functions for RAG Chat

# for deployment and UI
import streamlit as st
import os

# directories
data = './data/'
media = './media/'

# A template for the dialoque
template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""

# for loading and parcing web pages
from langchain_community.document_loaders import SeleniumURLLoader


# for reading and parcing multimodal pdf
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.utils.constants import PartitionStrategy


# for help of open-source LLMs
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate



# for text pre-processing
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore

# Web
# load page
def load_web_page(url):
    loader = SeleniumURLLoader(urls=[url])
    documents = loader.load()
    return documents


# parse the page text
def split_web_text(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    data = text_splitter.split_documents(docs)
    return data


def store_web_docs(docs, vector_store):
    vector_store.add_documents(docs)
    return
    

# PDF
# load pdf
def upload_pdf(file):
    with open(data + file.name, "wb") as f:
        f.write(file.getbuffer())


# extend llm with images
def text_from_image(file_path):
    print(file_path)
    model_with_image_context = llm.bind(images=[file_path])
    return model_with_image_context.invoke("Tell me what do you see in this picture.")


# parse pdf content
def parse_pdf(file_path, media):
    # settings
    elements = partition_pdf(
        file_path,
        strategy=PartitionStrategy.HI_RES,
        extract_image_block_types=["Image", "Table"],
        extract_image_block_output_dir=media
    )

    # extract the pdf text
    text_elements = [element.text for element in elements if element.category not in ["Image", "Table"]]
    print(text_elements)
    
    # extract the text from the images in the pdf document
    for file_path in os.listdir(media):
        fname, extension = os.path.splitext(file_path)
        if extension == 'jpg':
            image_text = text_from_image(media + file)
            text_elements.append(image_text)

    return "\n\n".join(text_elements)

def split_pdf_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_splitter.split_text(text)

def store_pdf_docs(texts, vector_store):
    vector_store.add_texts(texts)
    return

# both
def retrieve_docs(query, vector_store):
    return vector_store.similarity_search(query)

def answer_question(question, documents, llm):
    context = "\n\n".join([doc.page_content for doc in documents])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm
    return chain.invoke({"question": question, "context": context})

