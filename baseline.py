### Create baseline for the project
import json
from utils import read_cupt
import numpy as np

def model(train_sentences):
    baseline = {}

    for i, sentence in enumerate(train_sentences):
        baseline[i] = []

        for j, token in enumerate(sentence):
            if token["parseme:mwe"] != "*" and any((j != k) & (sentence[k]["parseme:mwe"] != "*") for k in range(max(0, j-2), min(len(sentence), j+3))):
                baseline[i].append((token["lemma"], j))

    sentences = [value for value in baseline.values() if value]
    
    final_list = []
    
    for sentence in sentences:
        i = 0
        while i < len(sentence):
            index = sentence[i][1]
            exp = [sentence[i][0]]
            i += 1
            for w2 in sentence[i:]:
                if w2[1] <= index + 2:
                    exp.append(w2[0])
                    index = w2[1]
                    i += 1
                else:
                    break
            if exp not in final_list :
                final_list.append(exp)

    return final_list

def evaluate_baseline(test, train) :
    score = 0 
    for exp in test :
        if exp in train :
            score += 1
    
    return score / len(test)

if __name__ == "__main__" :
    train_sentences = read_cupt("Dataset/FR/train.cupt")
    test_sentences = read_cupt("Dataset/FR/test.cupt")
    train = model(train_sentences)
    test = model(test_sentences)
    score = evaluate_baseline(train, test)
    print("Score : ", score)