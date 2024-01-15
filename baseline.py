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

        # On ajoute un label pour les mots B et I 
        for idx_word in labels: 
            lemma = sentence[idx_word]["lemma"]
            if lemma in words_B: 
                labels[idx_word] = 1
            elif lemma in words_I: 
                labels[idx_word] = 2

        # On complète les labels pour les mots G ou O
        for idx_word in range(len(sentence)):
            label = labels[idx_word]

            if label is not None: 
                continue

            # Début de la phrase est forcément O si ce n'est pas un B ou I
            if idx_word == 0: 
                labels[idx_word] = 0
                continue
            
            # Fin de la phrase est forcément O si ce n'est pas un B ou I
            if idx_word == (len(sentence))-1: 
                labels[idx_word] = 0
                continue 
            
            # Si le mot à gauche est B ou le mot à droite est I alors le label est G
            if labels[idx_word-1] == 1 or labels[idx_word+1] == 2:
                labels[idx_word] = 3
            # Sinon le label est O
            else: 
                labels[idx_word] = 0

        list_labels = list(labels.values())
        # return list_labels
        
        list_labels_IGO_1 = [1 if x == 2 else x for x in list_labels]
        return [2 if x == 3 else x for x in list_labels_IGO_1]

def evaluate_baseline(expression_test : list, expression_train : list) :
    score = 0 
    for expression in expression_test :
        if expression in expression_train :
            score += 1
    
    return score / len(expression_test)

if __name__ == "__main__" :
    import pandas as pd
    import ast
    import numpy as np
    from sklearn.metrics import f1_score
    from tqdm import tqdm

    train_sentences = read_cupt("Dataset/train.cupt")
    test_sentences = read_cupt("Dataset/test.cupt")

    baseline_model = BaselineModel()
    baseline_model.train(train_sentences)

    test = pd.read_csv("test_IGO.csv", sep="\t", converters={"token_list": ast.literal_eval,
                                                         "lemmas": ast.literal_eval,
                                                         'labels': ast.literal_eval})
    
    def calculate_scores_by_class(y_true, y_pred):
        # Filter out cases where y_true is -100
        valid_indices = [i for i, label in enumerate(y_true) if label != -100]
        y_true_valid = np.array([y_true[i] for i in valid_indices])
        y_pred_valid = np.array([y_pred[i] for i in valid_indices])

        # Calculate F1 score by class
        f1_scores = f1_score(y_true_valid, y_pred_valid, average=None, labels=np.unique(y_true_valid))

        return f1_scores

    y_pred = []
    y_true = []
    for i, sentence in tqdm(enumerate(test_sentences)) :
        y_predi = baseline_model.predict(sentence)
        y_true_ = test["labels"][i]

        y_pred.extend(y_predi)
        y_true.extend(y_true_)
    
    f_scores = calculate_scores_by_class(y_true, y_pred)
    
    print(f_scores)