# Create a WordCloud object: the text of all Docs
# Import the wordcloud library
from wordcloud import WordCloud 
import pandas as pd

def wordCloud(df):    
    long_string = [','.join(list(x)) for x in df['page_content'].values]
    longstring = str(long_string).replace('page_content','')
    longstring = str(longstring).replace('\\n',' ')
    # get stopwords
    stopw = loadlang(longstring)[1]
    # remove stopwords
    words = [word for word in longstring.split() if word.lower() not in stopw]
    clean_text = " ".join(words)
    # settings
    wordcloud = WordCloud(background_color="white", max_words=1500, contour_width=3, contour_color='steelblue')
    # view
    wordcloud.generate(str(clean_text))
    im = wordcloud.to_image()
    return im, longstring
