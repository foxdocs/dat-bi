import langdetect
from langdetect import DetectorFactory, detect, detect_langs
import spacy

def myLangDetect(text):
    mylang = ''
    mylangprob = 0.0
    try:
        langs = langdetect.detect_langs(text)
        mylang, mylangprop = langs[0].lang, langs[0].prob 
        
        # English
        if lang=='en': 
            models = ['en_core_web_md', 'da_core_news_md']
            default_model = 'en_core_web_md'
        # Danish    
        elif lang=='da': 
            models = ['da_core_news_md', 'en_core_web_md']
            default_model = 'da_core_news_md'
        # both    
        nlp = spacy.load(default_model)
        stopw = nlp.Defaults.stop_words
    
    # another language
    except langdetect.lang_detect_exception.LangDetectException:
        log.debug('Language not recognised')
        
    return default_model, stopw
