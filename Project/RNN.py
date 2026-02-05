import torch.nn as nn

class RNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=4096,
            hidden_size=256,
            num_layers=1,
            batch_first=True
        )
        self.fc = nn.Linear(256, 22)
        self.apply(self._init_weights)

    def forward(self, x):
        out, _ = self.rnn(x)

        out = self.fc(out[:, -1, :])


        return out

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
            nn.init.zeros_(m.bias)

        elif isinstance(m, (nn.RNN, nn.LSTM, nn.GRU)):
            for name, param in m.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)
