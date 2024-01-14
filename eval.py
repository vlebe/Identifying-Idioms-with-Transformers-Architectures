import numpy as np
from sklearn.metrics import f1_score
import torch
from transformers import DistilBertTokenizer, DistilBertModel 
from dataset2 import MWEDataset  
from torch.utils.data import DataLoader
from transformer import BertMWE
from tqdm import tqdm

def calculate_scores_by_class(y_true, y_pred):
    # Filter out cases where y_true is -100
    valid_indices = [i for i, label in enumerate(y_true) if label != -100]
    y_true_valid = np.array([y_true[i] for i in valid_indices])
    y_pred_valid = np.array([y_pred[i] for i in valid_indices])

    # Calculate F1 score by class
    f1_scores = f1_score(y_true_valid, y_pred_valid, average=None, labels=np.unique(y_true_valid))

    return f1_scores

def evaluate(model, data_loader, device, viterbi=False):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in tqdm(data_loader):
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            # Forward pass
            predicted = model.predict(input_ids=input_ids, attention_mask=attention_mask, viterbi_bool=viterbi)

            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())

    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    f1_scores = calculate_scores_by_class(y_true, y_pred)

    return f1_scores

if __name__ == "__main__" :
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
    bert_model = DistilBertModel.from_pretrained('distilbert-base-multilingual-cased')

    test_dataset = MWEDataset("test_IGO.csv", tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=16)

    model = BertMWE(3, bert_model, device)
    model.load_state_dict(torch.load("training_results/exp4/bert_mwe.pth"))
    model.to(device)

    print(evaluate(model, test_loader, device, viterbi=True))