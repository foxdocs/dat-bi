# LangChan file loader
import os
import os.path

from langchain_community.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import TextLoader

from pathlib import Path
# st.write(Path.cwd())

def loadFile(filename): 
    if filename.endswith('.pdf'):
        loader = PyPDFLoader(filename)
    elif filename.endswith('.doc') or filename.endswith('.docx'):
        loader = Docx2txtLoader(filename)
    elif filename.endswith('.txt'):
        loader = TextLoader(filename)
    docs = loader.load()
    return docs