import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from tqdm import tqdm
from task01 import WaterDataset
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
from torchmetrics import F1Score
import torch.nn.init as init
import pandas as pd

torch.manual_seed(42)

def train_model(dataloader_train, dataloader_val, optimizer, net, num_epochs, create_plot=False):
    
    criterion = nn.MSELoss()
    loss_history = []

    for epoch in tqdm(range(num_epochs)):
        train_loss = 0
        val_loss = 0


        for features, labels in dataloader_train:
            optimizer.zero_grad() # clear the gradients
            outputs = net(features) # forward pass
            loss = criterion(outputs, labels.view(-1, 1)) # calculate the loss
            loss.backward() # compute the gradients
            optimizer.step() # tweak weights
            train_loss += loss.item()
        
        train_loss /= len(dataloader_train)

        with torch.no_grad():
            for features, labels in dataloader_val:
                outputs = net(features)
                loss = criterion(outputs, labels.view(-1, 1))
                val_loss += loss.item()

        val_loss /= len(dataloader_val)
        
        loss_history.append((train_loss, val_loss))


    if create_plot:
        losses = np.array(loss_history)
        epochs = np.arange(num_epochs)

        plt.scatter(epochs, losses[:, 0], label="Train Loss")
        plt.scatter(epochs, losses[:, 1], label="Validation Loss")       
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.ylim(0, np.max(loss_history) * 1.05)
        plt.show()
    return loss_history

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(9, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)

        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(8)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
            nn.init.zeros_(m.bias)


    def forward(self, x):
        x = F.elu(self.bn1(self.fc1(x)))
        x = F.elu(self.bn2(self.fc2(x)))
        # x = F.elu(self.fc1(x))
        # x = F.elu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x
    

    

def main():

    filename = 'model_report_task04.xlsx'

    train_data = WaterDataset("water_train.csv")
    test_data = WaterDataset("water_test.csv")

    train_size = int(0.8 * len(train_data))
    val_size = len(train_data) - train_size

    train_dataset, val_dataset = random_split(
        train_data,
        [train_size, val_size]
    )

    #I increased the batch size, because when it was 2,
    #BN only messed up the training process
    dataloader_train = DataLoader(train_dataset, batch_size=32, shuffle=True)
    dataloader_val = DataLoader(val_dataset, batch_size=32, shuffle=True)
    dataloader_test = DataLoader(test_data, batch_size=32, shuffle=True)

    results = []

    best_model = None

    learning_rates = np.logspace(-6, -1, 10)
    num_epochs = 50

    for learning_rate in learning_rates:
        net = Net()
        optimizer = optim.AdamW(net.parameters(), lr = learning_rate)
        loss_history = train_model(dataloader_train, dataloader_val,optimizer, net, num_epochs)

        f1 = F1Score(task='binary')
        net.eval()

        with torch.no_grad():
            for features, labels in dataloader_test:
                outputs = net(features)
                preds = (outputs >= 0.5).float()
                f1(preds, labels.view(-1, 1))

        f1 = f1.compute()
        print(f'F1" {f1}')

        results.append({
            "learning_rate": learning_rate,
            "final_train_loss": loss_history[-1][0],
            "final_val_loss": loss_history[-1][1],
            "f1_score": f1.item()
        })

        if best_model == None or f1.item() > best_model['f1_score']:
            best_model = ({
                "learning_rate": learning_rate,
                "final_train_loss": loss_history[-1][0],
                "final_val_loss": loss_history[-1][1],
                "f1_score": f1.item()
            })

    learning_rate = best_model['learning_rate']
    num_epochs = 500

    net = Net()
    optimizer = optim.AdamW(net.parameters(), lr = learning_rate)
    loss_history = train_model(dataloader_train, dataloader_val,optimizer, net, num_epochs, True)

    f1 = F1Score(task='binary')
    net.eval()

    with torch.no_grad():
        for features, labels in dataloader_test:
            outputs = net(features)
            preds = (outputs >= 0.5).float()
            f1(preds, labels.view(-1, 1))

    f1 = f1.compute()
    print(f'F1" {f1}')

    results.append({
        "learning_rate": learning_rate,
        "final_train_loss": loss_history[-1][0],
        "final_val_loss": loss_history[-1][1],
        "f1_score": f1.item()
    })

    df = pd.DataFrame(results)
    df.to_excel("experiment_results.xlsx", index=False)

if __name__ == '__main__':
    main()