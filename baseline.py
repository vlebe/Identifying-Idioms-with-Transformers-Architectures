### Create baseline for the project
import re
import numpy as np
from utils import read_cupt

from conllu.models import SentenceList

pattern = r"\d:[A-Za-z]{3}"

def model(list_sentences : SentenceList):
    list_mwe = []
    
    for _, sentence in enumerate(list_sentences): 
        current_mwes = {}
        for token in sentence:
            id_mwe = token["parseme:mwe"]
            if id_mwe == "*": 
                continue
            elif re.match(pattern, id_mwe): 
                current_mwes[id_mwe[0]] = [token["form"]]
            elif id_mwe in current_mwes.keys():
                current_mwes[id_mwe].append(token["form"])
        if current_mwes != {}:
            list_mwe.append(list(current_mwes.values())[0])

    while [] in list_mwe:
        list_mwe.remove([])

    return list_mwe

def evaluate_baseline(expression_test : list, expression_train : list) :
    score = 0 
    for expression in expression_test :
        if expression in expression_train :
            score += 1
    
    return score / len(expression_test)

if __name__ == "__main__" :
    train_sentences = read_cupt("Dataset/FR/train.cupt")
    test_sentences = read_cupt("Dataset/FR/test.cupt")
    train = model(train_sentences)
    test = model(test_sentences)
    score = evaluate_baseline(train, test)
    print("Score : ", score)