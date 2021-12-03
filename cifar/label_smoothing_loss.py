import torch
import torch.nn as nn

class LabelSmoothingLoss(nn.Module):
    def __init__(self, prediction_size:int, smoothing:torch.tensor)->None:
        super(LabelSmoothingLoss, self).__init__()
        self.prediction_size = prediction_size
        assert (len(smoothing.shape) == 1 and smoothing.shape[0] == 1) or (len(smoothing.shape) == 1 and smoothing.shape[0] == prediction_size) or (len(smoothing.shape) == 2 and smoothing.shape[0] == prediction_size and smoothing.shape[1] == prediction_size)
        self.smoothing = nn.Parameter(smoothing.clone().detach())

    def forward(self, z:torch.tensor, y:torch.tensor)->None:
        nls = -z.log_softmax(dim=1)
        if len(self.smoothing.shape) == 1:
            if self.smoothing.shape[0] == 1:
                s = torch.sigmoid(self.smoothing)
                w = s/(z.shape[1]-1)*torch.ones_like(z)
                w[torch.arange(0,y.shape[0]),y] = (1.0-s)*torch.ones_like(y) #Broadcasting bug?!
            else:
                s = torch.sigmoid(self.smoothing[y])
                w = s.unsqueeze(1).repeat((1,10))/(z.shape[1]-1)
                w[torch.arange(0,y.shape[0]),y] = 1.0-s
        else:
            w = torch.softmax(torch.sigmoid(self.smoothing[y]), dim=1)
        return (w*nls).sum(dim=1).mean(dim=0)