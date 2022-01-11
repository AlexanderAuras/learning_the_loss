import torch

class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, prediction_size: int, smoothing: torch.tensor) -> None:
        super(LabelSmoothingLoss, self).__init__()
        self.prediction_size = prediction_size
        self.smoothing = torch.nn.Parameter(smoothing.clone().detach())

    def forward(self, z: torch.tensor, y: torch.tensor) -> None:
        nls = -z.log_softmax(dim=1)
        if len(self.smoothing.shape) == 1:
            if self.smoothing.shape[0] == 1:
                a = torch.sigmoid(self.smoothing)
                w = a/(z.shape[1]-1)*torch.ones_like(z)
                w[torch.arange(0,y.shape[0]),y] = (1.0-a)*torch.ones_like(y)
            else:
                tmp = torch.sigmoid(self.smoothing.repeat(y.shape[0],1))
                w = tmp.clone()
                tmp = torch.softmax(tmp, dim=1)
                w[torch.arange(0,y.shape[0]),y] = (1.0-tmp[torch.arange(0,y.shape[0]),y])
                w = torch.softmax(w, dim=1)
                '''
                smoothing:
                [3, 2, 3, 4, 2, 3, 3]
                tmp:
                [[0.3, 0.2, 0.3, 0.4, 0.2, 0.3, 0.3]
                 [0.3, 0.2, 0.3, 0.4, 0.2, 0.3, 0.3]]
                w:
                [[0.3, 0.2, 0.3, 0.4, 0.2, 0.3, 0.3]
                 [0.3, 0.2, 0.3, 0.4, 0.2, 0.3, 0.3]]
                tmp:
                [[0.15, 0.1, 0.15, 0.2, 0.1, 0.15, 0.15]
                 [0.15, 0.1, 0.15, 0.2, 0.1, 0.15, 0.15]]
                w: (y=[1,3])
                [[0.3, 0.9, 0.3, 0.2, 0.2, 0.3, 0.3]
                 [0.3, 0.1, 0.3, 0.8, 0.2, 0.3, 0.3]]
                w: (y=[1,3])
                [[0.3/3.1, 0.9/3.1, 0.3/3.1, 0.2/3.1, 0.2/3.1, 0.3/3.1, 0.3/3.1] 
                 [0.3/3.1, 0.1/3.1, 0.3/3.1, 0.8/3.1, 0.2/3.1, 0.3/3.1, 0.3/3.1] ]
                '''
        else:
            w = torch.softmax(self.smoothing[y], dim=1)
        return (w*nls).sum(dim=1).mean(dim=0)