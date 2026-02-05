import torch.nn as nn
import torch.nn.functional as F

class VGGNet(nn.Module):

    def __init__(self):
        super(VGGNet, self).__init__()
        self.fc1 = nn.Conv2d(in_channels=3,
                             out_channels=32,
                             kernel_size=3,
                             padding=1)
        # self.fc2 = nn.Conv2d(in_channels=64,
        #                      out_channels=64,
        #                      kernel_size=3,
        #                      padding=1)
        self.fc3 = nn.MaxPool2d(kernel_size=2)

        self.fc4 = nn.Conv2d(in_channels=32,
                             out_channels=64,
                             kernel_size=3,
                             padding=1)

        # self.fc5 = nn.Conv2d(in_channels=128,
        #                      out_channels=128,
        #                      kernel_size=3,
        #                      padding=1)

        self.fc6 = nn.MaxPool2d(kernel_size=2)

        self.fc7 = nn.Conv2d(in_channels=64,
                             out_channels=128,
                             kernel_size=3,
                             padding=1)

        # self.fc8 = nn.Conv2d(in_channels=256,
        #                      out_channels=256,
        #                      kernel_size=3,
        #                      padding=1)

        # self.fc9 = nn.Conv2d(in_channels=256,
        #                      out_channels=256,
        #                      kernel_size=3,
        #                      padding=1)

        self.fc10 = nn.MaxPool2d(kernel_size=2)

        # self.fc11 = nn.Conv2d(in_channels=256,
        #                       out_channels=512,
        #                       kernel_size=3,
        #                       padding=1)

        # self.fc12 = nn.Conv2d(in_channels=512,
        #                       out_channels=512,
        #                       kernel_size=3,
        #                       padding=1)

        # self.fc13 = nn.Conv2d(in_channels=512,
        #                       out_channels=512,
        #                       kernel_size=3,
        #                       padding=1)

        self.fc14 = nn.MaxPool2d(kernel_size=2)

        # self.fc15 = nn.Conv2d(in_channels=512,
        #                       out_channels=512,
        #                       kernel_size=3,
        #                       padding=1)

        # self.fc16 = nn.Conv2d(in_channels=512,
        #                       out_channels=512,
        #                       kernel_size=3,
        #                       padding=1)

        # self.fc17 = nn.Conv2d(in_channels=512,
        #                       out_channels=512,
        #                       kernel_size=3,
        #                       padding=1)

        self.fc18 = nn.MaxPool2d(kernel_size=2)

        #because input is 128x128 and not 224x224
        self.fc19 = nn.Linear(131072, 22)
        # self.fc20 = nn.Linear(2 * 512, 22)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc3(x)

        x = F.relu(self.fc4(x))
        # x = F.relu(self.fc5(x))
        x = self.fc6(x)

        x = F.relu(self.fc7(x))
        # x = F.relu(self.fc8(x))
        # x = F.relu(self.fc9(x))
        x = self.fc10(x)

        # x = F.relu(self.fc11(x))
        # x = F.relu(self.fc12(x))
        # x = F.relu(self.fc13(x))
        x = self.fc14(x)

        # x = F.relu(self.fc15(x))
        # x = F.relu(self.fc16(x))
        # x = F.relu(self.fc17(x))
        x = self.fc18(x)


        x = x.reshape(-1,)
        # x = F.relu(self.fc19(x))
        x = self.fc19(x)
        x = x.reshape(16, -1)
        print(x.shape)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

