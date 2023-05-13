import os
import re

import sklearn.metrics.pairwise
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def parse_sgm_file(filename):
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


def parse_sgm_directory(directory_path):
    """
    Парсинг всіх файлів формату SGM в директорії і повернення списку текстів.
    """
    texts = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".sgm"):
            file_path = os.path.join(directory_path, filename)
            text = parse_sgm_file(file_path)
            texts += text
    return texts


def additional_text_prep_vectorize(text_list):
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
    par_text = vectorizer.fit(only_texts)
    for text in text_list:
        text['body'] = vectorizer.transform([text['body']])
    return par_text


def body_vectorize(text_list):
    vectorizer = TfidfVectorizer()
    for text in text_list:
        text['body'] = vectorizer.fit_transform([text['body']])
    return text_list
