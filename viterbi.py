import torch
import pandas as pd
import numpy as np
import ast

def get_initial_proba(csv_file_path, igo=False) :
    df = pd.read_csv(csv_file_path, sep="\t", converters={"token_list": ast.literal_eval,
                                                         "lemmas": ast.literal_eval,
                                                         'labels': ast.literal_eval})

    labels = df.labels
    counter = 0
    for label in labels : 
        if label[0] != 0 :
            counter += 1
    if igo :
        return np.array([(len(df) - counter) / len(df), counter / len(df), 0])
    return np.array([(len(df) - counter) / len(df), counter / len(df), 0, 0])

def get_transition_proba(csv_file_path, igo=False) :
    df = pd.read_csv(csv_file_path, sep="\t", converters={"token_list": ast.literal_eval,
                                                         "lemmas": ast.literal_eval,
                                                         'labels': ast.literal_eval})

    labels = df.labels

    if igo :
        transtion_proba = {"O" : {"O" : 0,
                                    "I" : 0,
                                    "G" : 0},
                            "I" : {"O" : 0,
                                    "I" : 0,
                                    "G" : 0},
                            "G" : {"O" : 0,
                                    "I" : 0,
                                    "G" : 0},}
        
        class_mapping = {0 : "O",
                        1 : "I",
                        2 : "G"}
    
    else :
        transtion_proba = {"O" : {"O" : 0,
                                "B" : 0,
                                "I" : 0,
                                "G" : 0},
                            "B" : {"O" : 0,
                                "B" : 0,
                                "I" : 0,
                                "G" : 0},
                            "I" : {"O" : 0,
                                "B" : 0,
                                "I" : 0,
                                "G" : 0},
                            "G" : {"O" : 0,
                                "B" : 0,
                                "I" : 0,
                                "G" : 0},}

        class_mapping = {0 : "O",
                        1 : "B",
                        2 : "I",
                        3 : "G"}

    for label in labels : 
        for i in range(len(label)-1):
            transtion_proba[class_mapping[int(label[i])]][class_mapping[int(label[i+1])]] += 1

    raw_prob = np.array(pd.DataFrame(transtion_proba)).T
    sommes_lignes = np.sum(raw_prob, axis=1)

    return raw_prob / sommes_lignes[:, np.newaxis]

def viterbi(emission_matrices, initial_proba, transition_proba):
    B, N, T = emission_matrices.shape

    delta = torch.zeros(B, N, T)
    psi = torch.zeros(B, N, T)

    # Initialisation
    delta[:, :, 0] = emission_matrices[:, :, 0] * initial_proba.view(1, -1)

    # Calcul des différentes valeurs de delta
    for t in range(1, T):
        for j in range(N):
            liste = [
                delta[b, i, t-1] * transition_proba[i, j] * emission_matrices[b, j, t]
                for b in range(B)
                for i in range(N)
            ]
            liste = torch.tensor(liste).view(B, N)
            delta[:, j, t], psi[:, j, t] = liste.max(dim=1)

    # Détermination du meilleur chemin
    z_T = delta[:, :, T - 1].argmax(dim=1)
    z = torch.zeros(B, T, dtype=torch.long)
    z[:, T - 1] = z_T

    for t in range(T - 2, -1, -1):
        z[:, t] = psi[torch.arange(B), z[:, t + 1], t + 1]

    return z

# def viterbi(emission_matrix, initial_proba, transition_proba) :
#     shape = emission_matrix.shape

#     delta = torch.zeros(shape)
#     psi = torch.zeros(shape)

#     # Initialisation
#     delta[:, 0] = emission_matrix[:, 0] * initial_proba

#     # Calcul des différentes valeurs de delta
#     for t in range(0, shape[1] - 1) :
#         for j in range(shape[0]) :
#             liste = [ delta[i, t] * transition_proba[i, j] * emission_matrix[j, t + 1] for i in range(shape[0])]
#             delta[j, t + 1], psi[j, t + 1] = max(liste), torch.argmax(torch.tensor(liste))

#     # Détermination du meilleur chemin 
#     z_T = torch.argmax(torch.tensor([delta[i, shape[1]-1] for i in range(shape[0])])).long()
#     z = torch.zeros(shape[1]).long()
#     z[shape[1]-1] = z_T
#     for t in range(shape[1]-2, 0, -1) :
#         z[t] = psi[z[t+1], t+1]
    
#     return z

if __name__ == "__main__" :
    # # O B I G
    # initial_proba = torch.tensor([14332/14540, 208/14540, 0, 0])
    # initial_proba = torch.tensor([0.5, 0.5, 0, 0])
    # transition_proba = torch.ones(4, 4)
    # transition_proba[0, 2] = 0
    # transition_proba[0, 3] = 0
    # transition_proba[3, 0] = 0

    initial_proba = get_initial_proba("train_BIGO.csv")
    transition_proba = get_transition_proba("train_BIGO.csv")

    # initial_proba = np.array([0.5, 0.5, 0, 0])
    # transition_proba = np.array([[0.5, 0.5, 0, 0],
    #                             [0.25, 0.25, 0.25, 0.25],
    #                             [0.25, 0.25, 0.25, 0.25],
    #                             [0, 0.333333, 0.333333, 0.3333333]])

    print("Initial proba : ", initial_proba)
    print("Transition proba : ", transition_proba)

    input = ['le', 'chat', 'adore', 'le', "lait", '.']
    emission_proba = torch.rand(len(input), 4).T
    emission_proba[:, 0] = torch.tensor([1, 0, 0, 0])
    emission_proba[:, 1] = torch.tensor([0, 1, 0, 0])
    emission_proba[:, 2] = torch.tensor([0, 0, 0, 1])
    emission_proba[:, 3] = torch.tensor([0, 0, 1, 0])
    emission_proba[:, 4] = torch.tensor([1, 0, 0, 0])
    emission_proba[:, 5] = torch.tensor([1, 0, 0, 0])

    print("Emission proba : ", emission_proba)

    emission_sum = emission_proba.sum(dim=0)
    emission_proba = emission_proba / emission_sum

    print(emission_proba)

    print(viterbi(emission_proba, initial_proba, transition_proba))