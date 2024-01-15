import pandas as pd
import torch
from torch.utils.data import Dataset
import ast
from tqdm import tqdm

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
        labels = torch.cat((torch.tensor([-100]), labels, torch.tensor([-100])))

        encoding = self.tokenizer.encode_plus(tokens, add_special_tokens=True, 
                                              max_length=512, padding='max_length', 
                                              return_attention_mask=True, 
                                              return_tensors='pt', truncation=True)

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # Pad labels to size 128
        labels = torch.nn.functional.pad(labels, (0, 512 - labels.size(0)), value=-100)
        
        return input_ids, attention_mask, labels
    
class MWEDatasetEmbedding(Dataset):
    def __init__(self, csv_file_path, tokenizer, emb_path):
        self.df = pd.read_csv(csv_file_path, sep="\t", 
                              converters={"token_list": ast.literal_eval, 
                                          "lemmas": ast.literal_eval, 
                                          'labels': ast.literal_eval})
        self.embeddings = torch.load(emb_path)
        self.labels = self.create_labels(self.df, tokenizer)

    @staticmethod
    def create_labels(df, tokenizer):
        def mapping_tokens(tokens : list[str]): 
            list_mapping, counter, is_dash, is_parenthesis, is_digits, count_dash = [0], 0, False, False, False, 0
            for idx, token in enumerate(tokens):
                if not(token.startswith("##")):
                    if token == '[UNK]' :
                        token = "'"
                    elif token == "hui" and tokens[idx-1] == "'":
                        continue
                    
                    elif "'" in token: 
                        is_dash = False
                    elif is_dash:
                        if token in ("ce", "on"): 
                            counter += 1
                            is_dash = False
                        elif token == "-":
                            count_dash += 1
                            is_dash=True
                        else:
                            is_dash=False
                            count_dash = 0

                        if count_dash == 2 :
                            counter +=1 
                            is_dash = False
                            count_dash = 0
                            list_mapping[-1] += 1

                    elif "-" in token:
                        if idx == 0 :
                            counter += 1
                        elif tokens[idx-1] == "a":
                            counter += 1
                            is_dash = True
                            count_dash += 1
                        else : 
                            is_dash = True
                            count_dash += 1
                            
                    elif token in ("(", ")"): 
                        if token == "(": 
                            token_after = tokens[idx+1]
                            if len(token_after) >= 2: 
                                counter += 1
                            elif token_after.isdigit():
                                counter += 1
                            else : 
                                is_parenthesis = True
                        else: 
                            token_before = tokens[idx-1]
                            if len(token_before) >= 2: 
                                counter += 1
                            is_parenthesis = False
                    elif is_parenthesis: 
                        is_parenthesis=False

                    elif token.isdigit():
                        if not is_digits:
                            counter += 1
                            is_digits = True
                    elif is_digits:
                        if not (token == "," and tokens[idx+1].isdigit()) :
                            is_digits = False
                            counter += 1

                    elif token == '.' :
                        if tokens[idx-1] in ("etc", "cf", '.'):
                            continue
                        else :
                            counter += 1

                    else :
                        counter += 1
                list_mapping.append(counter)
            return list_mapping
        
        index_list = []
        for idx, row in tqdm(df.iterrows()): 
            tokens = row.token_list

            sentence = " ".join(tokens)
            tokens_bert = tokenizer.tokenize(sentence)

            list_mapping = mapping_tokens(tokens_bert)

            if len(row.labels) != max(list_mapping) or len(tokens_bert) > 512:
                continue

            index_list.append(idx)

            if idx%2000 == 0: 
                if idx > 1 :
                    torch.save(index_list2, f"input_id_{idx}.pt")
                index_list2 = torch.tensor([idx]).unsqueeze(0)
            else : 
                index_list2 = torch.cat((index_list2, torch.tensor([idx]).unsqueeze(0)), dim=0)

        torch.save(index_list2, f"input_id_fin.pt")
        torch.save(torch.tensor(index_list), "index_list_test.pt")

        return df.iloc[index_list].labels
    
    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        emb = self.embeddings[idx]
        lab_index = emb[512][0].int().item()
        labels = torch.tensor(self.labels.iloc[lab_index]).long()
        # labels = torch.tensor(self.labels.iloc[idx]).long()
        labels = torch.cat((torch.tensor([-100]), labels, torch.tensor([-100])))
        
        # Pad labels to size 128
        labels = torch.nn.functional.pad(labels, (0, 512 - labels.size(0)), value=-100)
        
        return emb[:512, :], labels
    
if __name__ == "__main__" :
    from transformers import DistilBertTokenizer

    dataset = MWEDatasetEmbedding('test_BIGO.csv', DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased'), "embeddings/test/embeddings_tensor_test_IGO_2000.pt")
    print(len(dataset.labels))
    print(dataset[0][0].shape, dataset[0][1].shape)