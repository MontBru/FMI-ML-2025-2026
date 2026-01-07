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
import torch.nn.init as init
from task02 import train_model

torch.manual_seed(42)

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
    
    learning_rate  = .001
    num_epochs = 1000

    train_data = WaterDataset("water_train.csv")
    test_data = WaterDataset("water_test.csv")

    #I increased the batch size, because when it was 2,
    #BN only messed up the training process
    dataloader_train = DataLoader(train_data, batch_size=32, shuffle=True)
    dataloader_test = DataLoader(test_data, batch_size=32, shuffle=True)

    net = Net()
    optimizer = optim.AdamW(net.parameters(), lr = learning_rate)
    loss_history = train_model(dataloader_train, optimizer, net, num_epochs, True)

    f1 = F1Score(task='binary')
    net.eval()

    with torch.no_grad():
        for features, labels in dataloader_test:
            outputs = net(features)
            preds = (outputs >= 0.5).float()
            f1(preds, labels.view(-1, 1))

    f1 = f1.compute()
    print(f'F1" {f1}')

    #Which of the following statements is true about batch normalization?
    #Response: C

    #The new f1 score is 0.5614973306655884

if __name__ == '__main__':
    main()