### Create baseline for the project
import re
import numpy as np
from utils import read_cupt

from conllu.models import SentenceList, TokenList

pattern = r"\d:[A-Za-z]{3}"

class BaselineModel(): 
    def __init__(self) -> None:
        self.list_mwe = []

    def train(self, list_sentences : SentenceList): 
        for _, sentence in enumerate(list_sentences): 
            current_mwes = {}
            for token in sentence:
                id_mwe = token["parseme:mwe"]
                if id_mwe == "*": 
                    continue
                elif re.match(pattern, id_mwe): 
                    current_mwes[id_mwe[0]] = [token["lemma"]]
                elif id_mwe in current_mwes.keys():
                    current_mwes[id_mwe].append(token["lemma"])
            if current_mwes != {}:
                self.list_mwe.append(list(current_mwes.values())[0])

        while [] in self.list_mwe:
            self.list_mwe.remove([])

    def predict(self, sentence : TokenList):
        list_first_words = [mwe[0] for mwe in self.list_mwe]

        current_mwes = []
        verified_mwes = []
        word_to_search = {}

        for token in sentence: 
            lemma = token["lemma"]
            if lemma in list_first_words: 
                #On cherche l'expression correspondant à lemma
                idx = list_first_words.index(lemma)
                mwe = self.list_mwe[idx]

                #On stocke l'expression comme expression en cours
                current_mwes.append(idx)

                #On définit le mot suivant de l'expression à chercher dans le texte
                if len(mwe): 
                    verified_mwes.append(mwe)
                else : 
                    word_to_search[idx] = mwe[1]

            elif lemma in list(word_to_search.values()):
                #On cherche l'expression correspondant à lemma
                idx_search = list(word_to_search.keys())[list(word_to_search.values()).index(lemma)]

                #On récupère l'expression
                mwe = self.list_mwe[idx_search]

                #Si l'expression est finie on met l'expression dans verified_mwes
                if mwe.index(lemma) == len(mwe)-1:
                    verified_mwes.append(mwe)

                #Sinon on ajoute le mot suivant dans les mots à chercher 
                else:
                    new_word_to_search = mwe[mwe.index(lemma)+1]
                    word_to_search[idx_search] = new_word_to_search
        
        labels = {idx : None for idx in range(len(sentence))}
        current_mwes = []
        words_B = [mwe[0] for mwe in verified_mwes]
        words_I = [words for mwe_verif in verified_mwes for words in mwe_verif]
        for word in words_B: 
            words_I.remove(word)

        for idx_word in labels: 
            lemma = sentence[idx_word]["lemma"]
            if lemma in words_B: 
                labels[idx_word] = 1
            elif lemma in words_I: 
                labels[idx_word] = 2

        for idx_word in range(len(sentence)):
            if idx_word == 0: 
                if labels[idx_word] is None:
                    labels[idx_word] = 0
                continue
            
            if idx_word == (len(sentence))-1: 
                if labels[idx_word] is None:
                    labels[idx_word] = 0
                continue 

            if labels[idx_word-1] == 1 or labels[idx_word+1] == 2:
                labels[idx_word] = 3
            else: 
                labels[idx_word] = 0

        list_labels = list(labels.values())
        return list_labels
        
        list_labels_IGO_1 = [1 if x == 2 else x for x in list_labels]
        return [2 if x == 3 else x for x in list_labels_IGO_1]

def evaluate_baseline(expression_test : list, expression_train : list) :
    score = 0 
    for expression in expression_test :
        if expression in expression_train :
            score += 1
    
    return score / len(expression_test)

if __name__ == "__main__" :
    train_sentences = read_cupt("Dataset/FR/train.cupt")
    test_sentences = read_cupt("Dataset/FR/test.cupt")

    baseline_model = BaselineModel()
    baseline_model.train(train_sentences)

    test = model(test_sentences)
    score = evaluate_baseline(train, test)
    print("Score : ", score)