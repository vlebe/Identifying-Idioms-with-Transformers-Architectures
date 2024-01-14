import torch
from viterbi import viterbi, get_initial_proba, get_transition_proba
import torch.nn.functional as F
    
class BertMWE(torch.nn.Module):
    def __init__(self, num_labels, model, device):  
        super().__init__()
        self.initial_proba = torch.tensor(get_initial_proba("train_IGO.csv", igo=True))
        self.transition_proba = torch.tensor(get_transition_proba("train_IGO.csv", igo=True))
        self.bert = model
        self.device = device
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(768, 384),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(384, num_labels)
        )

    def forward(self, input_ids, attention_mask=None):
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = bert_output[0]
        return self.classifier(pooled_output)
    
    def predict(self, input_ids, attention_mask=None, viterbi_bool=False):
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = bert_output[0]
        output = self.classifier(pooled_output)
        
        if viterbi_bool:
            output = output.transpose(1, 2)
            output = F.softmax(output, dim=1).cpu()
            viterbi_preds = viterbi(output, self.initial_proba, self.transition_proba)
            
            return viterbi_preds.long()
        
        _, predicted = torch.max(output, dim=2)

        return predicted
    
class BertMWEEmbedding(torch.nn.Module):
    def __init__(self, num_labels, device):  
        super().__init__()
        self.initial_proba = torch.tensor(get_initial_proba("train_IGO.csv", igo=True))
        self.transition_proba = torch.tensor(get_transition_proba("train_IGO.csv", igo=True))
        self.device = device
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(768, 384),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(384, num_labels)
        )

    def forward(self, emb_input):
        return self.classifier(emb_input)
    
    def predict(self, emb_input, viterbi_bool=False):
        output = self.classifier(emb_input)
        
        if viterbi_bool:
            output = output.transpose(1, 2)
            output = F.softmax(output, dim=1).cpu()
            viterbi_preds = viterbi(output, self.initial_proba, self.transition_proba)
            
            return viterbi_preds.long()
        
        _, predicted = torch.max(output, dim=2)

        return predicted