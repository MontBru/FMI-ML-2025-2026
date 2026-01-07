import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from tqdm import tqdm
from task01 import WaterDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from torchmetrics import F1Score

torch.manual_seed(42)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(9, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x

def train_model(dataloader_train, optimizer, net, num_epochs, create_plot=False):
    
    criterion = nn.MSELoss()
    loss_history = []

    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0
        for features, labels in dataloader_train:
            optimizer.zero_grad() # clear the gradients
            outputs = net(features) # forward pass
            loss = criterion(outputs, labels.view(-1, 1)) # calculate the loss
            loss.backward() # compute the gradients
            optimizer.step() # tweak weights
            epoch_loss += loss.item()

        loss_history.append(epoch_loss/num_epochs)


    if create_plot:
        plt.scatter(np.arange(num_epochs), loss_history)
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.ylim(0, max(loss_history) * 1.05)
        plt.show()
    return loss_history


def main():
    
    learning_rate  = .001
    num_epochs = 10

    optimizers = [
        ("SGD", optim.SGD),
        ("RMSprop",optim.RMSprop),
        ("Adam",optim.Adam),
        ("AdamW",optim.AdamW)
    ]

    best_loss_for_optimizer = {}
    
    train_data = WaterDataset("water_train.csv")
    test_data = WaterDataset("water_test.csv")
    dataloader_train = DataLoader(train_data, batch_size=2, shuffle=True)
    dataloader_test = DataLoader(test_data, batch_size=2, shuffle=True)

    for name, optimizer in optimizers:
        net = Net()
        optimizer = optimizer(net.parameters(), lr = learning_rate)
        loss_history = train_model(dataloader_train, optimizer, net, num_epochs)
        best_loss = min(loss_history)
        best_loss_for_optimizer[name] = best_loss

    print(best_loss_for_optimizer)

    #Comparing the losses for each optimizer we can see
    #That the order from worst to best is:
    #   SGD -> RMSprop -> Adam -> AdamW

    net = Net()
    optimizer = optim.AdamW(net.parameters(), lr = learning_rate)
    loss_history = train_model(dataloader_train, optimizer, net, 1000, True)

    f1 = F1Score(task='binary')
    net.eval()

    with torch.no_grad():
        for features, labels in dataloader_test:
            outputs = net(features)
            preds = (outputs >= 0.5).float()
            f1(preds, labels.view(-1, 1))

    f1 = f1.compute()
    print(f'F1" {f1}')

    #Should have been in a Jupyter Notebook but it took a long time
    #to train, so here are the results:

    # {'SGD': 18.121064932644366, 'RMSprop': 18.037152986228467, 'Adam': 17.543199684470892, 'AdamW': 16.74258538223803}
    # 100%|████████████████████████████████████████████| 1000/1000 [06:13<00:00,  2.68it/s]
    # F1" 0.5508021116256714

    #The model isn't very good, it couldn't overfit the data even with
    #1000 epochs and it stopped getting better early so this indicates
    #that probably there is a better model.
    


if __name__ == "__main__":
    main()