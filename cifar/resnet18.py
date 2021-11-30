from residual_block import ResidualBlock
from torch import nn

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3), nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),
            ResidualBlock( 64,  64, nn.Conv2d( 64,  64, 3, 1, 1), nn.BatchNorm2d( 64), nn.ReLU(), nn.Conv2d( 64,  64, 3, 1, 1), nn.BatchNorm2d( 64)),
            ResidualBlock( 64,  64, nn.Conv2d( 64,  64, 3, 1, 1), nn.BatchNorm2d( 64), nn.ReLU(), nn.Conv2d( 64,  64, 3, 1, 1), nn.BatchNorm2d( 64)),
            ResidualBlock( 64, 128, nn.Conv2d( 64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(), nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128)),
            ResidualBlock(128, 128, nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(), nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128)),
            ResidualBlock(128, 256, nn.Conv2d(128, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.ReLU(), nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256)),
            ResidualBlock(256, 256, nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(), nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256)),
            ResidualBlock(256, 512, nn.Conv2d(256, 512, 3, 2, 1), nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512)),
            ResidualBlock(512, 512, nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(), nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512)),
            nn.AvgPool2d(7),
            nn.Flatten(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        return self.layers(x)