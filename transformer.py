from transformers import DistilBertTokenizer, DistilBertModel
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
distilbert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')

class DistilBertForMWE(torch.nn.Module):
    def __init__(self, num_labels):  
        super().__init__()
        self.distilbert = distilbert_model

        self.classifier = torch.nn.Linear(distilbert_model.config.dim,num_labels )

    def forward(self, input_ids, attention_mask=None):
        distilbert_output = self.distilbert(input_ids, attention_mask=attention_mask)
        hidden_state = distilbert_output[0]  # (batch_size, sequence_length, hidden_size)
        pooled_output = hidden_state[:, 0]   
        logits = self.classifier(pooled_output)
        return logits
    
model = DistilBertForMWE(num_labels=2)
