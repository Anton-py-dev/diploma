import os
import re

import sklearn.metrics.pairwise
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class SGMReader:
    def parse_sgm_file(self, filename):
        with open(filename, 'r') as f:
            soup = BeautifulSoup(f, 'html.parser')
            texts = []
            for text in soup.find_all('text'):
                title = text.find('title').text if text.find('title') else None
                dateline = text.find('dateline').text if text.find('dateline') else None
                b = text.find('body').text if text.find('body') else None
                body = re.sub('[^0-9a-zA-Z]+', ' ', str(b)).lower()
                textex = {'title': title, 'dateline': dateline, 'body': body}
                texts.append(textex)
            return texts

    def parse_sgm_directory(self, directory_path):
        """
        Парсинг всіх файлів формату SGM в директорії і повернення списку текстів.
        """
        texts = []
        for filename in os.listdir(directory_path):
            if filename.endswith(".sgm"):
                file_path = os.path.join(directory_path, filename)
                text = self.parse_sgm_file(file_path)
                texts += text
        return texts
