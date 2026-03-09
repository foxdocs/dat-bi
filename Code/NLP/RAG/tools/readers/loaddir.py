# LangChain Document reader
from langchain_community.docstore.document import Document
from langchain_community.document_loaders import DirectoryLoader

def loadDir(path, filetype='*', recursive = True):    
    loader = DirectoryLoader(path, glob="**/*." + filetype, show_progress=True)
    docs = loader.load()
    return docs