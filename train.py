import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from dataset2 import MWEDataset
from transformer import BertMWE
from transformers import DistilBertTokenizer, DistilBertModel
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from eval import evaluate, calculate_scores_by_class
import matplotlib.pyplot as plt
import numpy as np
from utils import EarlyStopping
import argparse
import os

def train(model, train_loader, val_loader, optimizer, loss, epochs, early_stopping, device, save_dir, scheduler):
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
        
        scheduler.step(total_val_loss)
        early_stopping(total_val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping stopped")
            break

    torch.save(model.state_dict(), save_dir + "bert_mwe.pth")
    return train_losses, val_losses

def main(args):

    if not os.path.exists(args.save_dir) :
        os.mkdir(args.save_dir)

    files = os.listdir(args.save_dir)

    save_dir = args.save_dir + "/exp" + str(len(files)) + '/' 
    os.makedirs(save_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
    bert_model = DistilBertModel.from_pretrained('distilbert-base-multilingual-cased')

    print("Loading datasets...")
    train_dataset = MWEDataset(f"train_{args.output_encoding}.csv", tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)

    val_dataset = MWEDataset(f"val_{args.output_encoding}.csv", tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    test_dataset = MWEDataset(f"test_{args.output_encoding}.csv", tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    model = BertMWE(args.n_classes, bert_model, device)

    # Freeze Camembert parameters
    for param in model.bert.base_model.parameters():
        param.requires_grad = False

    model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.2)

    # IOG (3.202401664041446, 97.56650403112943, 99.23109430482913)
    # BIGO (2.1647543052400664, 98.92802365281834, 98.61239387711518, 99.22285181764475)
    loss = CrossEntropyLoss(reduction='none', weight=torch.tensor([3.2, 98.9, 98.6]).to(device))
    early_stopping = EarlyStopping(path=save_dir + "best_weights.pth")

    train_losses, val_losses = train(model, train_loader, val_loader, optimizer, loss, args.epochs, early_stopping, device, save_dir, scheduler)

    model.load_state_dict(torch.load(save_dir + "best_weights.pth"))
    model.to(device)
    f1_scores = evaluate(model, test_loader, device)

    print('Test F1 scores : ', f1_scores)
    
    with open(save_dir + "results.txt", 'a') as f:
        f.write("Bert MWE\n")
        f.write("Test F1 scores : " + str(f1_scores) + "\n")

    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.legend(["Train loss", "Val loss"])
    plt.savefig(save_dir + "bert_mwe_training.png")

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save-dir', type=str, default="training_results")
    parser.add_argument('--n_classes', type=int, default=4)
    parser.add_argument("-o", '--output-encoding', type=str, default="BIGO")

    main(parser.parse_args())