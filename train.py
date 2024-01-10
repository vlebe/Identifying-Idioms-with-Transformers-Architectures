import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from dataset2 import MWEDataset
from transformer import DistilBertForMWE
from transformers import DistilBertTokenizer, DistilBertModel
from dataset2 import collate_batch

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
distilbert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')

learning_rate = 5e-5
epochs = 3
batch_size = 32


train_dataset = MWEDataset("/Users/meliya/Desktop/PSTALN/projet/PSTALN/test_IGO.csv", tokenizer)
train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_batch)


model = DistilBertForMWE(num_labels=2)
model.train()


optimizer = AdamW(model.parameters(), lr=learning_rate)


for epoch in range(epochs):
    total_loss = 0
    for batch in train_loader:
        input_ids, attention_mask, labels = batch

        # Afficher les dimensions des tenseurs d'entrée
        print(f"Taille des input_ids: {input_ids.size()}")
        print(f"Taille des attention_mask: {attention_mask.size()}")

        # Remise à zéro des gradients
        optimizer.zero_grad()

        # Propagation avant
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Propagation arrière et optimisation
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader)}")


# Sauvegarde du modèle
torch.save(model.state_dict(), "distilbert_mwe.pth")
