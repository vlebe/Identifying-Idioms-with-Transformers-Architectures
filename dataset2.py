import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer
import ast
from torch.nn.utils.rnn import pad_sequence

class MWEDataset(Dataset):
    def __init__(self, csv_file_path, tokenizer):
        self.df = pd.read_csv(csv_file_path, sep="\t", 
                              converters={"token_list": ast.literal_eval, 
                                          "lemmas": ast.literal_eval, 
                                          'labels': ast.literal_eval})
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        tokens = self.df["token_list"].iloc[idx]
        labels = torch.tensor(self.df["labels"].iloc[idx]).long()
        encoding = self.tokenizer(tokens, return_tensors='pt', 
                                  padding='max_length', truncation=True, 
                                  max_length=128)
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)

        print(f"input_ids length: {input_ids.size(1)}, attention_mask length: {attention_mask.size(1)}")
        return input_ids, attention_mask, labels
        
