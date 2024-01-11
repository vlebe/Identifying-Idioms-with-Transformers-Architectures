import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from dataset import MWEDataset
from transformer import CamembertMWE
from transformers import CamembertTokenizer, CamembertModel, DistilBertTokenizer, DistilBertModel
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import numpy as np

def train(model, train_loader, val_loader, optimizer, loss, epochs, device):
    train_losses = []
    val_losses = []

    print(f"Training on {device}...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader):
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            batch_loss = 0
            for i in range(outputs.shape[0]):
                mask = (labels[i] != -100).float().to(device)
                batch_loss += (loss(outputs[i], labels[i]) * mask).mean()

            batch_loss /= outputs.shape[0]
            total_loss += batch_loss.item()

            batch_loss.backward()
            optimizer.step()
        
        total_loss /= len(train_loader)

        if epoch % 5 == 0:
            torch.save(model.state_dict(), f"camembert_mwe2_{epoch}.pth")

        model.eval()
        with torch.no_grad():
            total_val_loss = 0
            y_true = []
            y_pred = []
            for batch in val_loader :
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                _, predicted = torch.max(outputs, dim=2)
                y_true.extend(labels.tolist())
                y_pred.extend(predicted.tolist())

                batch_loss = 0
                for i in range(outputs.shape[0]):
                    mask = (labels[i] != -100).float().to(device)
                    batch_loss += (loss(outputs[i], labels[i]) * mask).mean()

                batch_loss /= outputs.shape[0]

                total_val_loss += batch_loss.item()
        total_val_loss /= len(val_loader)

        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()

        f1_scores = calculate_scores_by_class(y_true, y_pred)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {total_loss}, Val Loss: {total_val_loss}")
        train_losses.append(total_loss)
        val_losses.append(total_val_loss)

        print("Val scores : ", f1_scores)

    torch.save(model.state_dict(), "camembert_mwe2.pth")
    return train_losses, val_losses

def calculate_scores_by_class(y_true, y_pred):
    # Filter out cases where y_true is -100
    valid_indices = [i for i, label in enumerate(y_true) if label != -100]
    y_true_valid = np.array([y_true[i] for i in valid_indices])
    y_pred_valid = np.array([y_pred[i] for i in valid_indices])

    # Calculate F1 score by class
    f1_scores = f1_score(y_true_valid, y_pred_valid, average=None, labels=np.unique(y_true_valid))

    return f1_scores

def evaluate(model, data_loader, device):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs, dim=2)

            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())

    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    f1_scores = calculate_scores_by_class(y_true, y_pred)

    return f1_scores

if __name__ == '__main__' :
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
    # bert_model = CamembertModel.from_pretrained('camembert-base')

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
    bert_model = DistilBertModel.from_pretrained('distilbert-base-multilingual-cased')

    learning_rate = 1e-4
    epochs = 30
    batch_size = 16

    print("Loading datasets...")
    train_dataset = MWEDataset("train_BIGO.csv", tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)

    val_dataset = MWEDataset("val_BIGO.csv", tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    test_dataset = MWEDataset("test_BIGO.csv", tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = CamembertMWE(4, bert_model, device)

    # Freeze Camembert parameters
    for param in model.bert.base_model.parameters():
        param.requires_grad = False

    model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    loss = CrossEntropyLoss(reduction='none', weight=torch.tensor([3.2, 98.9, 98.6, 99.2]).to(device))

    train_losses, val_losses = train(model, train_loader, val_loader, optimizer, loss, epochs, device)

    f1_scores = evaluate(model, test_loader, device)
    
    with open("results2.txt", 'a') as f:
        f.write("Camembert MWE\n")
        f.write("Test F1 scores : " + str(f1_scores) + "\n")

    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.legend(["Train loss", "Val loss"])
    plt.savefig("camembert_mwe_training2.png")