import numpy as np
from sklearn.metrics import f1_score
import torch

def calculate_scores_by_class(y_true, y_pred):
    # Filter out cases where y_true is -100
    valid_indices = [i for i, label in enumerate(y_true) if label != -100]
    y_true_valid = np.array([y_true[i] for i in valid_indices])
    y_pred_valid = np.array([y_pred[i] for i in valid_indices])

    # Calculate F1 score by class
    f1_scores = f1_score(y_true_valid, y_pred_valid, average=None, labels=np.unique(y_true_valid))

    return f1_scores

def evaluate(model, data_loader, device):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs, dim=2)

            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())

    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    f1_scores = calculate_scores_by_class(y_true, y_pred)

    return f1_scores