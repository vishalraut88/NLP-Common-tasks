import pandas as pd
import numpy as np
import re
import sys
sys.path.append("..")
from preprocessing.Preprocessing import Textpreprocessing
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class clustering():

    def __init__(self):
        pass
    
    def tfidf_vectorization(self,data):
        
        vect=TfidfVectorizer()
        x = vect.fit_transform(data)
        return x
        
    
    def vectorization(self,vect_type,data):
        if(vect_type=="tfidf"):
            vectors = self.tfidf_vectorization(data)
            return vectors
    
    def find_best_k(self):
        return 5
    
    
    def KMeans_clustering(self,vect_type,data):
        #1. vectorize the data
        vectorized_data = self.vectorization(vect_type,data)
        #Optional step can be to perform Dimentional reduction like PCA
        
        #2. Find optimum value of k
        k=self.find_best_k()
        
        #3. perform clustering
        model =KMeans(n_clusters=k)
        result = model.fit(vectorized_data)  
        silhou_score = silhouette_score(vectorized_data,result.labels_,metric="euclidean")        
        df = pd.DataFrame(data.values,columns=["text"])
        df["label"]=model.labels_
        
        return df
        
        
    def lda(self,text_data):
        text_data = text_data.apply(lambda x : x.split())
