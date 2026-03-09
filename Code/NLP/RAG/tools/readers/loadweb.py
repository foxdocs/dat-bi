# LangChain load web page
from langchain_community.docstore.document import Document
from langchain_community.document_loaders import WebBaseLoader

def loadWeb(url):
    # loader = UnstructuredLoader(web_url=url)
    loader = WebBaseLoader(url)
    docs = loader.load()
    return docs