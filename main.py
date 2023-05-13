from klusterizerAdditionalFunc import *

if __name__ == '__main__':
    text = parse_sgm_file("testFiles/reut2-000.sgm")
    #textV = body_vectorize(text)
    textV = additional_text_prep_vectorize(text)
    #text = parse_sgm_directory("testFiles")
    print(type(textV))
    print(len(textV))

