from torch.utils.data import Dataset
import pandas as pd
import ast
import torch

class MWEDataset(Dataset):
    def __init__(self, csv_file_path):
        self.df = pd.read_csv(csv_file_path, sep="\t", converters={"token_list": ast.literal_eval,
                                                         "lemmas": ast.literal_eval,
                                                         'labels': ast.literal_eval})
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        return torch.tensor(self.df["labels"].iloc[idx]).long(), self.df["token_list"].iloc[idx]
    
if __name__ == "__main__" :
    dataset = MWEDataset("train_IGO.csv")
    print(dataset[1])