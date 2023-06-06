import os
import re
from bs4 import BeautifulSoup


class SGMReader:
    def parse_sgm_file(self, filename):
        """
        Парсинг одного файлу .sgm
        :param filename: шлях до файлу
        :return: список текстів з файлу
        """
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
        Парсинг всіх файлів в дерикторії з розширенням .sgm
        :param directory_path: шлях до папки
        :return: список текстів з файлів
        """
        texts = []
        for filename in os.listdir(directory_path):
            if filename.endswith(".sgm"):
                file_path = os.path.join(directory_path, filename)
                text = self.parse_sgm_file(file_path)
                texts += text
        return texts

    def parse_sgm_file_list(self, file_list):
        """
        Парсинг файлів зі списку що мають розширення .sgm
        :param file_list: список шляхів до файлів
        :return: список текстів з файлів
        """
        texts = []
        for filename in file_list:
            file_path = os.path.join(filename)
            text = self.parse_sgm_file(file_path)
            texts += text
        return texts