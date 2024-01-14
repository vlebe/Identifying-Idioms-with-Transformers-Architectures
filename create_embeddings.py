import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer, DistilBertModel
import ast
import numpy as np
from tqdm import tqdm
class MWEDataset(Dataset):
    def __init__(self, csv_file_path, tokenizer, embedding_model):
        self.df = pd.read_csv(csv_file_path, sep="\t", 
                              converters={"token_list": ast.literal_eval, 
                                          "lemmas": ast.literal_eval, 
                                          'labels': ast.literal_eval})
        self.tokenizer = tokenizer
        self.embedding_model = embedding_model
        self.embeddings_tensor = self.create_embeddings(self.df, tokenizer, embedding_model)

    @staticmethod
    def create_embeddings(df : pd.DataFrame, tokenizer, embedding_model): 
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

        def mean_embed(list_mapping : list[int], embeddings : torch.Tensor):
            current_map, list_current_embed, embeddings_final = None, None, None
            for idx, mapping in enumerate(list_mapping): 

                if idx==0: 
                    current_map = mapping
                    list_current_embed = [embeddings[idx, :, :]]

                elif idx==len(list_mapping)-1:
                    if current_map == mapping:
                        list_current_embed.append(embeddings[idx, :, :])
                    else: 
                        mean_embeddings = None
                        for idx_embed, embed in enumerate(list_current_embed): 
                                if idx_embed==0: 
                                    mean_embeddings = embed
                                else : 
                                    mean_embeddings += embed
                        new_line = mean_embeddings/len(list_current_embed)
                        embeddings_final = torch.cat((embeddings_final, new_line.detach().unsqueeze(0).to(device)), dim=0)
                        list_current_embed = [embeddings[idx, :, :]]
                    mean_embeddings = None
                    for idx_embed, embed in enumerate(list_current_embed): 
                            if idx_embed==0: 
                                mean_embeddings = embed
                            else : 
                                mean_embeddings += embed 
                    new_line = mean_embeddings/len(list_current_embed)
                    embeddings_final = torch.cat((embeddings_final, new_line.detach().unsqueeze(0).to(device), embeddings[idx+1:,:].to(device)), dim=0)

                elif current_map != mapping:
                    mean_embeddings = None
                    for idx_embed, embed in enumerate(list_current_embed): 
                            if idx_embed==0: 
                                mean_embeddings = embed
                            else : 
                                mean_embeddings += embed
                    if embeddings_final is not None:
                        new_line = mean_embeddings/len(list_current_embed)
                        embeddings_final = torch.cat((embeddings_final, new_line.detach().unsqueeze(0).to(device)), dim=0)
                    else: 
                        new_line = mean_embeddings/len(list_current_embed)
                        embeddings_final = torch.tensor(new_line.detach().to(device)).unsqueeze(0)

                    list_current_embed = [embeddings[idx, :, :]]
                    current_map = mapping

                else : 
                    list_current_embed.append(embeddings[idx, :, :])

            return embeddings_final

        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")
        
        index_list = []
        for idx, row in tqdm(df.iterrows()): 
            tokens = row.token_list

            sentence = " ".join(tokens)
            tokens_bert = tokenizer.tokenize(sentence)

            list_mapping = mapping_tokens(tokens_bert)

            if len(row.labels) != max(list_mapping) or len(tokens_bert) > 512:
                continue

            encoding = tokenizer.encode_plus(tokens, add_special_tokens=True, 
                                        max_length=512, padding='max_length', 
                                        return_attention_mask=True, 
                                        return_tensors='pt', truncation=True)
            input_ids, attention_mask = encoding['input_ids'], encoding['attention_mask']

            embedding = embedding_model(input_ids=input_ids, attention_mask=attention_mask)[0]
            new_line = torch.full((1, 1, 768), float(idx))

            # Concatenate along the second dimension (dimension 1)
            embedding = torch.cat((embedding, new_line), dim=1) 

            if idx%2000 == 0: 
                if idx > 1 :
                    torch.save(embeddings_tensor, f"embeddings_tensor_test_IGO_{idx}.pt")
                embeddings_tensor = embedding.detach().to(device)
            else : 
                embeddings_tensor = torch.cat((embeddings_tensor, embedding.detach().to(device)))

        return mean_embed(list_mapping, embeddings_tensor)

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
    
def main(): 
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
    bert_model = DistilBertModel.from_pretrained('distilbert-base-multilingual-cased')

    print(f"Loading dataset")
    train_dataset = MWEDataset("test_BIGO.csv", tokenizer, bert_model)
    torch.save(train_dataset.embeddings_tensor, "embeddings_tensor_test.pt")

if __name__=="__main__": 
    main()