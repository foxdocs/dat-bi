# LangChain Wikipedia Document readers

from langchain_community.document_loaders import WikipediaLoader
from langchain_community.docstore.document import Document

def loadWiki(query, lang='en', n=1):
    if query is not None:
        loader = WikipediaLoader(query=query, lang=lang, load_max_docs=n)
        docs = loader.load()
    else:
        ("Please, load a file.")
    return docs
