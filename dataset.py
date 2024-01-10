import pandas as pd
import torch
from torch.utils.data import Dataset
import ast

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
        tokens = self.df["lemmas"].iloc[idx]
        labels = torch.tensor(self.df["labels"].iloc[idx]).long()
        labels = torch.cat((torch.tensor([-100]), labels, torch.tensor([-100])))

        input = " ".join(tokens) 
        tokens = self.tokenizer.tokenize(input)

        encoding = self.tokenizer.encode_plus(tokens, add_special_tokens=True, 
                                              max_length=400, padding='max_length', 
                                              return_attention_mask=True, 
                                              return_tensors='pt', truncation=True)

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # Pad labels to size 128
        labels = torch.nn.functional.pad(labels, (0, 400 - labels.size(0)), value=-100)
        
        return input_ids, attention_mask, labels