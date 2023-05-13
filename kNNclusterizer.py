import os
import numpy as np
from bs4 import BeautifulSoup
from klusterizerAdditionalFunc import *


class kNNсlusterizer:
    def __init__(self, clusterNum=4, neighborNum=5):
        self.clusterNum = clusterNum
        self.neighborNum = neighborNum
        self.vocab = None


    def init_cluster(self, text_list):
        for text in text_list:
            pass

    def additional_text_prep_vectorize(self, text_list):
        lemmatizer = WordNetLemmatizer()
        nltk.download('stopwords')
        stop_words = list(stopwords.words('english'))
        vectorizer = TfidfVectorizer(stop_words=stop_words)
        only_texts = []
        for text in text_list:
            lem_text = [lemmatizer.lemmatize(word) for word in text['body'].split(" ")]
            final_text = ' '.join(lem_text)
            only_texts.append(final_text)
        # create vocab
        self.vocab = vectorizer.fit(only_texts)
        for text in text_list:
            text['body'] = vectorizer.transform([text['body']])
        return text_list