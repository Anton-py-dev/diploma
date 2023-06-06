import statistics
import numpy as np
import matplotlib.pyplot as plt
from klusterizerAdditionalFunc import SGMReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class kNNсlusterizer:
    def __init__(self, clusterNum=2, neighborNum=5, iterNum=2):
        self.clusterNum = clusterNum
        self.neighborNum = neighborNum
        self.iterNum = iterNum
        self.vocab = None
        self.clusterList = None
        self.nameList = None
        self.matrix = None
        self.nearNeigh = None
        self.textNum = 0

    @staticmethod
    def readSGM(path, folder=False, file=False, list=False):
        """
        Зчитування файлів з розширенням .sgm
        :param path: шлях до потрібних файлів/папки
        :param folder: True - якщо шлях веде до папки
        :param file: True - якщо шлях веде до файлу
        :param list: True - якщо заданий список шляхів до файлів файлів
        :return: список текстів з файлів
        """
        sgm = SGMReader()
        if folder:
            text_list = sgm.parse_sgm_directory(path)
        elif file:
            text_list = sgm.parse_sgm_file(path)
        elif list:
            text_list = sgm.parse_sgm_file_list(path)
        return text_list

    def init_cluster(self, text_list):
        """
        Перетворює список текстів в більш структурований вигляд,
        а саме список словників де для кожного тексту є поля з назвою,
        списком параметрів або термів із основної частини тексту а також присвоює
        випадковий кластер для кожного тексту
        :param text_list: список текстів
        :return: список словників
        """
        n_col, n_row = text_list[0]['body'].toarray().shape
        dt = np.dtype([('name', 'U10'), ('coordinates', np.ndarray, (n_col, n_row)), ('cluster', 'i4')])
        cluster_arr = np.empty(shape=[len(text_list), 1], dtype=dt)
        for i in range(len(text_list)):
            random_cluster = np.random.randint(self.clusterNum)
            cluster_arr['name'][i] = text_list[i]['title']
            cluster_arr['coordinates'][i] = text_list[i]['body'].toarray()
            cluster_arr['cluster'][i] = random_cluster
            # cluster_arr[i] = (text_list[i]['title'], text_list[i]['body'], random_cluster)
        return cluster_arr

    def one_matrix_cluster_init(self, text_list):
        """
        Створює три структури даних, дві з яких списоки з назвами і
        номером кластеру, а також матрицю з мараметрами або термами з тексту

        :param text_list: список текстів
        :return: матриця термів/па
        """
        n_col, n_row = text_list[0]['body'].toarray().shape
        self.textNum = len(text_list)
        self.matrix = np.empty((len(text_list), n_row))
        self.clusterList = np.empty(len(text_list), dtype="int")
        self.nameList = np.empty(len(text_list), dtype="<U16")
        for i in range(len(text_list)):
            random_cluster = i % self.clusterNum
            # random_cluster = np.random.randint(self.clusterNum)
            self.nameList[i] = text_list[i]['title']
            self.clusterList[i] = random_cluster
            self.matrix[i] = text_list[i]['body'].toarray()
        return self.matrix

    def one_matrix_init(self, text_list):
        n_col, n_row = text_list[0]['body'].toarray().shape
        self.matrix = np.empty((len(text_list), n_row))
        for i in range(len(text_list)):
            self.matrix[i] = text_list[i]['body'].toarray()
        return self.matrix

    @staticmethod
    def body_vectorize(text_list):
        """
        Векторизує список текстів
        :param text_list: список текстів
        :return: веторизований список текстів
        """
        vectorizer = TfidfVectorizer()
        for text in text_list:
            text['body'] = vectorizer.fit_transform([text['body']])
        return text_list

    def additional_text_prep_vectorize(self, text_list):
        """
        Функція виконує повну підготовку текстів до векторизації, а саме
        лематизацію, видалення стопслів та векторизує методом TF-IDF
        :param text_list: список текстів у вигляді словника зі вмістом
        під ключем "body"
        :return: Векторизований список текстів
        """
        nltk.download('wordnet')
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

    def visualise(self, matrix, reduction="TSNE"):
        """
        Відображає документи і їх взаямне розміщення у вигляді точок на площині
        для кращого розуміння як працює алгоритм
        :param matrix: матриця параметрів текстових документів
        :param reduction: метод зменшення розмірності матриці для візуалізації
        """
        if reduction == "TSNE":
            x = TSNE(n_components=2, perplexity=30, learning_rate=200)
            plt.title("TSNE")
        else:
            x = PCA(n_components=2)
            plt.title("PCA")

        X_reduced = x.fit_transform(matrix)
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=self.clusterList)
        plt.show()

    def build_nearest_matrix(self):
        """
        Будує матрицю найближчих сусідів
        :return: матриця найближчих сусідів
        """
        distances = cosine_similarity(self.matrix)
        self.nearNeigh = distances.argsort()[:, 1:self.neighborNum + 1]
        return self.nearNeigh

    def find_k_nearest_neighbors(self):
        """
        Обчислює найближчих сусідів до кожного елемента
        і на основі цього повертає список нових кластерів
        :return: список нових кластерів
        """
        neighbor_indices = self.build_nearest_matrix()
        future_clusters = np.empty(self.textNum, dtype="int")
        for i in range(self.textNum):
            near_clusters = [self.clusterList[ind] for ind in neighbor_indices[i]]
            mode_cluster = statistics.mode(near_clusters)
            future_clusters[i] = mode_cluster

        return future_clusters

    def clusterize(self):
        """
        Цикл кластеризації
        :return: None
        """
        i = 0
        while i < self.iterNum:
            i += 1
            future_clust = self.find_k_nearest_neighbors()
            if np.array_equal(future_clust, self.clusterList):
                self.visualise(self.matrix)
                return
            self.clusterList = future_clust
            self.visualise(self.matrix)

    def returnClusters(self):
        """
        Формує словник, де ключ це номер кластеру, а
        значення список з назвами документів, що відносяться
        до кластеру
        :return: словник кластерів
        """
        l = {}
        for i in range(len(self.clusterList)):
            if self.clusterList[i] in l.keys():
                l[self.clusterList[i]].append(self.nameList[i])
            else:
                l[self.clusterList[i]] = [self.nameList[i]]
        return l
