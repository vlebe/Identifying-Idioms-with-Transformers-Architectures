import pandas as pd
import torch
from torch.utils.data import Dataset
import ast
import numpy as np
from vocab import Vocab

class MWEDataset(Dataset):
    def __init__(self, csv_file_path, tokenizer):
        self.df = pd.read_csv(csv_file_path, sep="\t", 
                              converters={"token_list": ast.literal_eval, 
                                          "lemmas": ast.literal_eval, 
                                          'labels': ast.literal_eval})
        self.tokenizer = tokenizer
        self.vocab = Vocab(csv_file_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        tokens = self.df["token_list"].iloc[idx]
        labels = torch.tensor(self.df["labels"].iloc[idx]).long()
        labels = torch.cat((torch.tensor([-100]), labels, torch.tensor([-100])))

        # indices = []
        # final_tokens = []
        # for i, token in enumerate(tokens):
        #     bert_tokens = self.tokenizer.tokenize(token)
        #     final_tokens.extend(bert_tokens)
        #     indices.extend([i] * len(bert_tokens))

        encoding = self.tokenizer.encode_plus(tokens, add_special_tokens=True, 
                                              max_length=400, padding='max_length', 
                                              return_attention_mask=True, 
                                              return_tensors='pt', truncation=True)

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # Pad labels to size 400
        labels = torch.nn.functional.pad(labels, (0, 400 - labels.size(0)), value=-100)

        # Pad indices to size 400
        # indices = torch.nn.functional.pad(torch.tensor(indices), (0, 400 - len(indices)), value=-1)

        # Pad tokens to size 400
        # tokens = torch.cat((torch.tensor([-1]), tokens, torch.tensor([-1])))
        # for i, token in enumerate(tokens) :
        #     tokens[i] = self.vocab.get_word_index(token)

        # tokens = torch.nn.functional.pad(torch.tensor(tokens), (0, 400 - len(tokens)), value=-1)

        return input_ids, attention_mask, labels
        return input_ids, attention_mask, tokens, indices, labels