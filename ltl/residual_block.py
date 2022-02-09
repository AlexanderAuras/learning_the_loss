from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, *args: nn.Module):
        super(ResidualBlock, self).__init__()
        self.layers = nn.ModuleList(args)
        self.relu = nn.ReLU()
        self.channel_dim_matcher = None
        if in_channels != out_channels:
            self.channel_dim_matcher = nn.Conv2d(in_channels, out_channels, 1, 2)

    def forward(self, x):
        y = x
        for layer in self.layers:
            y = layer(y)
        if self.channel_dim_matcher != None:
            x = self.channel_dim_matcher(x)
        return self.relu(x+y)