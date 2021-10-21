from torch import nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        count_hidden = 100
        self.layers = nn.Sequential(
            nn.Linear(2, count_hidden),
            nn.ReLU(),
            nn.Linear(count_hidden, count_hidden),
            nn.ReLU(),
            nn.Linear(100, 2)
        )

    def forward(self, x):
        return self.layers(x)