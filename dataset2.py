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
        



def collate_batch(batch):
    input_ids, attention_masks, labels = zip(*batch)

    # d'abord on convertit les listes en tenseurs et les pad si nécessaire
    input_ids = [ids.clone().detach() for ids in input_ids]
    attention_masks = [mask.clone().detach() for mask in attention_masks]


    # icii on gère le cas où il y a une seule entrée dans une séquence
    input_ids = [ids if ids.ndim > 1 else ids.unsqueeze(0) for ids in input_ids]
    attention_masks = [mask if mask.ndim > 1 else mask.unsqueeze(0) for mask in attention_masks]

    # on pad les séquences pour qu'elles aient toutes la même longueur
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)

    return input_ids, attention_masks, labels



if __name__ == "__main__":
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    dataset = MWEDataset("train_IGO.csv", tokenizer)
    data_loader = DataLoader(dataset, batch_size=32, collate_fn=collate_batch)

    for input_ids, attention_mask, labels in data_loader:
        print(input_ids.shape, attention_mask.shape, labels.shape)



