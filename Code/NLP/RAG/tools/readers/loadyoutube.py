# LangChain Document Youtube video transcripts loader
from langchain_community.docstore.document import Document
from langchain_community.document_loaders import YoutubeLoader

def loadYoutube(url):
    # loader = YoutubeLoader.from_youtube_url(url, add_video_info=True, language = lang, translation = lang)
    loader = YoutubeLoader.from_youtube_url(url, add_video_info=False)
    docs = loader.load()
    return docs
