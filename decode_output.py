import pandas as pd
from utils import read_cupt, extract_numbers
import numpy as np

def create_bigo_csv(sentences, csv_file_name) :
    data_structure = {
                        'token_list': [],
                        'lemmas': [],
                        'labels': [],
                    }
    
    for i, sentence in enumerate(sentences):
        token_list = []
        lemmas = []
        labels = np.zeros(len(sentence))
        past_exp_ids = {}
        for j, word in enumerate(sentence) :
            token_list.append(word['form'])
            lemmas.append(word['lemma'])

            if word['parseme:mwe'] != "*" :
                exp_id = extract_numbers(word['parseme:mwe'])
                for id in exp_id :
                    if id not in past_exp_ids.keys() :
                        past_exp_ids[id] = [j]
                        labels[j] = 1
                    else :
                        past_exp_ids[id].append(j)
                        labels[j] = 2

        all_indices = set()
        for indices in past_exp_ids.values():
            all_indices.update(indices)

        for _, indices in past_exp_ids.items():
            for i in range(min(indices), max(indices) + 1):
                if i not in indices:
                    if i not in all_indices:
                        all_indices.add(i)
                        labels[i] = 3

        data_structure['token_list'].append(token_list)
        data_structure['lemmas'].append(lemmas)
        data_structure['labels'].append(labels)
    
    df = pd.DataFrame(data_structure)
    df.to_csv(csv_file_name, index=False)

def create_igo_csv(sentences, csv_file_name) :
    data_structure = {
                        'token_list': [],
                        'lemmas': [],
                        'labels': [],
                    }
    
    for i, sentence in enumerate(sentences):
        token_list = []
        lemmas = []
        labels = np.zeros(len(sentence))
        past_exp_ids = {}
        for j, word in enumerate(sentence) :
            token_list.append(word['form'])
            lemmas.append(word['lemma'])

            if word['parseme:mwe'] != "*" :
                exp_id = extract_numbers(word['parseme:mwe'])
                for id in exp_id :
                    if id not in past_exp_ids.keys() :
                        past_exp_ids[id] = [j]
                        labels[j] = 1
                    else :
                        past_exp_ids[id].append(j)
                        labels[j] = 1

        all_indices = set()
        for indices in past_exp_ids.values():
            all_indices.update(indices)

        for _, indices in past_exp_ids.items():
            for i in range(min(indices), max(indices) + 1):
                if i not in indices:
                    if i not in all_indices:
                        all_indices.add(i)
                        labels[i] = 2

        data_structure['token_list'].append(token_list)
        data_structure['lemmas'].append(lemmas)
        data_structure['labels'].append(labels)
    
    df = pd.DataFrame(data_structure)
    df.to_csv(csv_file_name, index=False)

if __name__ == "__main__" :
    train_sentences = read_cupt("Dataset/FR/train.cupt")
    test_sentences = read_cupt("Dataset/FR/test.cupt")
    val_sentences = read_cupt("Dataset/FR/dev.cupt")

    class_mapping = {
        "O" : 0,
        "B" : 1,
        "I" : 2,
        "G" : 3,
    }

    create_bigo_csv(train_sentences, "train_BIGO.csv")
    create_bigo_csv(test_sentences, "test_BIGO.csv")
    create_bigo_csv(val_sentences, "val_BIGO.csv")

    create_igo_csv(train_sentences, "train_IGO.csv")
    create_igo_csv(test_sentences, "test_IGO.csv")
    create_igo_csv(val_sentences, "val_IGO.csv")