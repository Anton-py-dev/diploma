import os
import numpy as np
from scipy.sparse import csr_matrix
from bs4 import BeautifulSoup
from klusterizerAdditionalFunc import SGMReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class kNNсlusterizer:
    def __init__(self, clusterNum=4, neighborNum=5):
        self.clusterNum = clusterNum
        self.neighborNum = neighborNum
        self.vocab = None

    @staticmethod
    def readSGM(path, folder=False, file=True):
        sgm = SGMReader()
        if folder:
            text_list = sgm.parse_sgm_directory(path)
        elif file:
            text_list = sgm.parse_sgm_file(path)
        return text_list

    def init_cluster(self, text_list):
        n_col, n_row = text_list[0]['body'].toarray().shape
        dt = np.dtype([('name', 'U10'), ('coordinates', np.ndarray, (n_col, n_row)), ('cluster', 'i4')])
        cluster_arr = np.empty(shape=[len(text_list), 1], dtype=dt)
        for i in range(len(text_list)):
            random_cluster = np.random.randint(self.clusterNum)
            cluster_arr['name'][i] = text_list[i]['title']
            cluster_arr['coordinates'][i] = text_list[i]['body'].toarray()
            cluster_arr['cluster'][i] = random_cluster
            #cluster_arr[i] = (text_list[i]['title'], text_list[i]['body'], random_cluster)
        return cluster_arr

    @staticmethod
    def body_vectorize(text_list):
        vectorizer = TfidfVectorizer()
        for text in text_list:
            text['body'] = vectorizer.fit_transform([text['body']])
        return text_list

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

