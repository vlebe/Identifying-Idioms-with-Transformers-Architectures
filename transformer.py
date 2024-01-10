import torch

class CamembertMWE(torch.nn.Module):
    def __init__(self, num_labels, model, device):  
        super().__init__()
        self.bert = model
        self.device = device
        self.classifier = torch.nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask=None):
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        hidden_state = bert_output[0]  
        pred = torch.zeros(hidden_state.shape[0], 400, 4).to(self.device)
        for i in range(hidden_state.shape[0]) :
            pooled_output = hidden_state[i]
            pred[i] = self.classifier(pooled_output)
        return pred