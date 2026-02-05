import torch.nn as nn
import torch.nn.functional as F

class KeypointNN(nn.Module):
  def __init__(self):
    super(KeypointNN, self).__init__()
    self.fc1 = nn.Conv2d(17 * 3, 100)
    self.fc2 = nn.Linear(100, 22)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x

  def _initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)