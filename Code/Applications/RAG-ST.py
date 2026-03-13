#!/usr/bin/env python
# coding: utf-8

import streamlit as st

from langchain_community.docstore.document import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import YoutubeLoader 
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import WikipediaLoader
# from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# for text pre-processing
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore

# for help of open-source LLMs
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

import os, pprint
from typing import List

# ## Load Documents
import tools
from tools import readers, utils
from tools.readers import *

def load_docs(url):
    if 'http' in url:
            if 'youtube' in url:
                docs = loadyoutube.loadYoutube(url);
            elif 'wikipedia' in url:
                docs = loadwiki.loadWiki(url);
            else:
                docs = loadweb.loadWeb(url);
    else:
            docs = loadfile.loadFile(url);
    return docs

def save_uploaded(documents, file_path):
    try:
        with open(file_path, 'w') as f:
            for docs in documents:
                f.write(f"{docs}\n")
        return True
    except Exception as e:
        print(f"Error occurred while writing the file: {e}")
        return False

def store_docs(documents):
    # create random indeces
    uuids = [str(uuid4()) for _ in range(len(documents))]
    # store the documents
    vector_store.add_documents(documents=documents, ids=uuids)
    return

# parse the collected text
def split_docs(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    # texts = text_splitter.split_text(documents)
    return chunks    

# Retrieve Data
def retrieve_docs(query, k):
    retrieved = vector_store.similarity_search(query, k)
    return retrieved

def retrieve_docs_score(query, k):
    retrieved = vector_store.similarity_search_with_score(query, k)
    return retrieved

def answer_question(question, documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm
    return chain.invoke({"question": question, "context": context})


st.set_page_config(
    page_title="BI Chat Bot",
    page_icon="🧊",
    layout="wide",
    # initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'mailto:tdi@ek.dk',
        'About': "BI DAT 2026"
    }
)


documents = []
url = 
finished = False

while not finished:
    url = input('Enter the location of your source: ')   
    finished = url == 'end'
    if finished: break;
    docs = load_docs(url)
    documents.extend(docs)

# for test
# /Users/tdi/Documents/GitHub/foxdocs/dat-bi/Data/what-is-rag.pdf
# https://www.youtube.com/watch?v=Y08Nn23o_mY&t=69s
# https://en.wikipedia.org/wiki/Retrieval-augmented_generation

len(documents)


save_uploaded(documents, './data/temp_save.resources')


# ## Split the Documents
chunks = split_docs(documents)
len(chunks)


# ## Embed and Store the Documents
# We encode (embed) the text chunks into digital vectors (an array of multiple numbers, representing the importance of the text components), then we store the vectors in a vector database.<br>
# We can use either external or in-memory database.

# We need a pretrained AI model for encoding the embeddings. We work with open-source models from Ollama or HuggingFace.

from langchain_chroma import Chroma
from uuid import uuid4

from langchain_ollama import OllamaEmbeddings

get_ipython().system('ollama list')

# choose Ollama model for embedding
model = "embeddinggemma:300m"
embeddings = OllamaEmbeddings(model=model)

# vector_store = InMemoryVectorStore(embeddings)

vector_store = Chroma(
    collection_name="rag",
    embedding_function=embeddings,
    persist_directory="./data/chroma_db"
)


store_docs(chunks)



# similarity search test
results = retrieve_docs(q, 3)

for res in results:
    print(f"* [{res.page_content}] [{res.metadata}]")

# similarity search test
results = retrieve_docs_score(q, 3)

for res, score in results:
    print(f"* [SIM={score:3f}][{res.metadata}]")


# ## Augment LLM Generation

get_ipython().system('ollama list')

llm = OllamaLLM(model = "qwen3.5:9b")
# llm = OllamaLLM(model = 'llama3.2:latest')

# A template for a dialoque
template = """
You are an assistant for BI/AI learning tasks. 
Use the following pieces of retrieved context to augment the LLM in answering the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""

# ## Apply RAG

question = 'What is RAG?'
# send the user's question to the vector db for retrieving the relevant context
retrieved = retrieve_docs(question, 2)

retrieved[0].metadata
answer = answer_question(question, retrieved)

answer