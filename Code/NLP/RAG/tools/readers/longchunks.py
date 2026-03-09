# splits text into chunks by specifying a separator character or string 
# chunk size is measured by number of characters, fixed size?

from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter, SpacyTextSplitter

def chunkText(what):  
    c_text_splitter = CharacterTextSplitter (
        chunk_size = 100,
        chunk_overlap  = 10,
        length_function = len,
        # separator = '\n'
    )
    splits = c_text_splitter.split_text(what)
    return splits 

# splits text recursively until chunks are under the specified size
# first, double \n,then \n, then " ", then ""
def chunkDocs(what):  
    r_text_splitter = RecursiveCharacterTextSplitter(
        # Set custom chunk size
        chunk_size = 100,
        chunk_overlap  = 10,
        # Use length of the text as the size measure
        # length_function = len,
        # is_separator_regex = True
        # Use only "\n\n" as the separator
        separators = ['\n\n', '\n', ' ', '']
    )
    splits = r_text_splitter.split_documents(what)
    # splits = r_text_splitter.split_text(what)
    return splits 

def chunkSpacy(what):
    s_text_splitter = SpacyTextSplitter (
        # Set custom chunk size
        chunk_size = 100,
        chunk_overlap  = 5,
        # Use length of the text as the size measure
        # length_function = len,
        # is_separator_regex = True
        # Use only "\n\n" as the separator
        # separators = ['\n\n', '\n', ' ', '']
    )
    splits = s_text_splitter.split_documents(what)
    return splits 
    
def chunkSplit(func, what):  
    # storing the function in a variable  
    return func(what)  
