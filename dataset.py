import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer, DistilBertModel
import ast
import numpy as np
from vocab import Vocab

class MWEDataset(Dataset):
    def __init__(self, csv_file_path, tokenizer, embedding_model):
        self.df = pd.read_csv(csv_file_path, sep="\t", 
                              converters={"token_list": ast.literal_eval, 
                                          "lemmas": ast.literal_eval, 
                                          'labels': ast.literal_eval})
        self.tokenizer = tokenizer
        self.embedding_model = embedding_model
        self.vocab = Vocab(csv_file_path)
        self.embeddings_tensor = self.create_embeddings(self.df, tokenizer, embedding_model)

    @staticmethod
    def create_embeddings(df : pd.DataFrame, tokenizer, embedding_model): 
        embeddings_tensor = torch.zeros(len(df), 400, 768)

        for idx, row in df.iterrows(): 
            tokens = row.token_list
            encoding = tokenizer.encode_plus(tokens, add_special_tokens=True, 
                                        max_length=400, padding='max_length', 
                                        return_attention_mask=True, 
                                        return_tensors='pt', truncation=True)
            input_ids, attention_mask = encoding['input_ids'].squeeze(0), encoding['attention_mask'].squeeze(0)

            embedding = embedding_model(input_ids=input_ids, attention_mask=attention_mask)[0]
            
            embeddings_tensor[idx, :, :] = embedding.detach()

            raise KeyboardInterrupt

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        tokens = self.df["token_list"].iloc[idx]
        labels = torch.tensor(self.df["labels"].iloc[idx]).long()
        labels = torch.cat((torch.tensor([-100]), labels, torch.tensor([-100])))

        embedding = self.embeddings_tensor[idx, :, :]
        
        # Pad labels to size 400
        labels = torch.nn.functional.pad(labels, (0, 400 - labels.size(0)), value=-100)

        # Pad indices to size 400
        # indices = torch.nn.functional.pad(torch.tensor(indices), (0, 400 - len(indices)), value=-1)

        # Pad tokens to size 400
        # tokens = torch.cat((torch.tensor([-1]), tokens, torch.tensor([-1])))
        # for i, token in enumerate(tokens) :
        #     tokens[i] = self.vocab.get_word_index(token)

        # tokens = torch.nn.functional.pad(torch.tensor(tokens), (0, 400 - len(tokens)), value=-1)

        return embedding, labels
        return input_ids, attention_mask, tokens, indices, labelsÒ
    
def main(): 
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
    bert_model = DistilBertModel.from_pretrained('distilbert-base-multilingual-cased')

    print(f"Loading dataset")
    train_dataset = MWEDataset("train_BIGO.csv", tokenizer, bert_model)

if __name__=="__main__": 
    main()