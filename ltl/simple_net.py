from torch import nn

class SimpleCIFARNet(nn.Module):
    def __init__(self):
        super(SimpleCIFARNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d( 3, 16, 3, 1, 1), 
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1, 1), 
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1), 
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), 
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16*16*128, 10)
        )

    def forward(self, x):
        return self.layers(x)