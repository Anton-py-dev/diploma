from kNNclusterizer import kNNсlusterizer
import nltk

if __name__ == '__main__':
    # Створення об'єкта класу кластеризатора
    k = kNNсlusterizer(neighborNum=2, clusterNum=2, iterNum=4)
    # Зчитування файлів
    textlist = k.readSGM(list=True, path=["testFiles/reut2-000.sgm", "testFiles/reut2-001.sgm"])
    # Векторизація текстів
    textVectorised = k.additional_text_prep_vectorize(textlist)
    # Початкова ініціалізація кластерів
    matrix = k.one_matrix_cluster_init(textVectorised)
    # Візуалізація початкової ініціалізації
    vis = k.visualise(matrix)
    # Кластеризація
    k.clusterize()
    # Повернення кластеризованих текстів у вигляді словника
    result = k.returnClusters()

