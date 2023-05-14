from klusterizerAdditionalFunc import *
from kNNclusterizer import kNNсlusterizer

if __name__ == '__main__':

    k = kNNсlusterizer()
    tl = k.readSGM(file=True, path="testFiles/reut2-000.sgm")
    textV = k.additional_text_prep_vectorize(tl)
    cl = k.init_cluster(textV)
    vis = k.visualise(cl)
    print(type(textV))
    print(len(textV))