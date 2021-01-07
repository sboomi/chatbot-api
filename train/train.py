import argparse
from pathlib import Path
from src.dataset import IntentLoader
from src.model import IntentClassifier
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import numpy as np

def fit(model, train_dl, epochs, device, criterion, optim, model_path):
    history_loss = []
    model, criterion = model.to(device), criterion.to(device)
    model.zero_grad()
    model.train()

    for epoch in trange(epochs, unit="epoch", desc="Train"):
        loss_batch = 0
        for bn, (data, label) in enumerate(train_dl):
            data, label = data.to(device), label.to(device)

            # Forward pass
            output = model(data)
            loss = criterion(output, label.argmax(dim=1))

            optim.zero_grad()
            loss.backward()
            optim.step()  

            # Register loss
            loss_batch += loss.detach().item()

        history_loss.append(loss_batch / len(train_dl))
        if (epoch+1)%10==0:
            print(f"Current loss: {loss_batch:.2f}")

    torch.save(model.state_dict(), model_path)

    return history_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-fp','--filepath' ,dest='fp', 
    default="train/data/intents.json", help="The path of your intents.json file")
    parser.add_argument('-sp', '--savepath', dest='sp', default='train/models')
    parser.add_argument('-bs','--batch-size', type=int, 
    dest='bs', default=5, help="the size of your total batches")
    parser.add_argument('-ep','--epochs', type=int, 
    dest='epochs', default=200, help='the number of epochs necessary for the training')
    
    args = parser.parse_args()

    epochs = args.epochs
    bs = args.bs

    print("Define paths")
    data_path = Path(args.fp)
    graph_path = Path('train/graphs')
    if not graph_path.exists():
        graph_path.mkdir()

    print("Checking model folder...")
    model_path = Path(args.sp)
    if not model_path.exists():
        model_path.mkdir()
        model_path /= 'chatbot_model_000'
    else:
        last_file = sorted([f.stem for f in model_path.iterdir()], reverse=True)[0]
        last_ver = int(last_file.split("_")[-1])
        model_path /= f'chatbot_model_{last_ver:03d}'

    print("Intents.json")
    il = IntentLoader(data_path)
    il.fit()
    il.describe()
    train_x, train_y = il.to_train_set()

    print("Dataset and Dataloader")
    train_ds = TensorDataset(torch.Tensor(train_x), torch.Tensor(train_y))
    print(f"N° of documents in dataset: {len(train_ds)}")

    train_dl = DataLoader(train_ds, batch_size=bs)
    print(f"N° of batches {len(train_dl)} (batch size {train_dl.batch_size})")

    # Model parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Current device: {device}")
    criterion = nn.CrossEntropyLoss()
    model = IntentClassifier(train_x, train_y)
    optim = torch.optim.SGD(model.parameters(), lr=0.01, 
    weight_decay=1e-6, momentum=0.9, nesterov=True)

    #Training
    loss_history = fit(model, train_dl, epochs, device, criterion, optim, model_path)

    # Create fig
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(np.array(loss_history))
    ax.set_title("Train loss")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    fig.savefig(graph_path / f'training_loss_{last_ver:03d}.png')

if __name__ == '__main__':
    main()