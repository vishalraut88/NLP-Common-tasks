import pandas as pd
import re
import string
import nltk


class Textpreprocessing():
    
    def __init__(self):
        pass
    
    def text_cleaning(self,text):
        
        """
        Description : This function performs text cleaning that includes following:
            - lower the case
            - remove tags
            - remove special characters
            - remove punctuation
            - remove stopwords
        input parameters : Text that needs to be cleaned.
        returns : cleaned text after pre processing
        """
        text = text.lower()
        text = re.sub("</?.*?>"," <> ",text)
        text = re.sub("(\\d|\\W)+"," ",text)
        text = text.replace("_","")
        #remove punctuation
        text = [char for char in text if char not in string.punctuation]
        text = ''.join(text)
        stopwords = nltk.corpus.stopwords.words('english')
        text = ' '.join([word for word in text.split() if word not in (stopwords)])            
        return text
