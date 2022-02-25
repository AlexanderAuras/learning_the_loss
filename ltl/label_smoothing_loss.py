import torch

class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, prediction_size: int, smoothing: torch.tensor, sigmoid: bool = True) -> None:
        super(LabelSmoothingLoss, self).__init__()
        self.prediction_size = prediction_size
        self.smoothing = torch.nn.Parameter(smoothing.clone().detach())
        self.sigmoid = sigmoid

    def forward(self, z: torch.tensor, y: torch.tensor) -> None:
        nls = -z.log_softmax(dim=1)
        if len(self.smoothing.shape) == 1:
            if self.smoothing.shape[0] == 1:
                if self.sigmoid:
                    a = torch.sigmoid(self.smoothing)
                else:
                    a = self.smoothing
                w = a/(z.shape[1]-1)*torch.ones_like(z)
                w[torch.arange(0,y.shape[0]),y] = (1.0-a)*torch.ones_like(y)
            else:
                if self.sigmoid:
                    tmp = torch.sigmoid(self.smoothing.repeat(y.shape[0],1))
                else:
                    tmp = self.smoothing.repeat(y.shape[0],1)
                tmp = torch.softmax(tmp, dim=1)
                w = torch.zeros_like(tmp)
                w[torch.arange(0,y.shape[0]),y] = (1.0-tmp[torch.arange(0,y.shape[0]),y])
                tmp2 = tmp.clone()
                tmp2[torch.arange(0,y.shape[0]),y] = torch.zeros_like(y, dtype=torch.float32)
                tmp2[tmp2!=0.0] = torch.exp(tmp2[tmp2!=0.0])
                tmp2 = tmp2/tmp2.sum(dim=1).unsqueeze(dim=1).repeat((1,tmp2.shape[1]))
                tmp2 = tmp2*(1.0-w[w!=0].unsqueeze(dim=1).repeat((1,tmp2.shape[1])))
                w += tmp2
        else:
            if self.sigmoid:
                print("[WARN]: Usage of matrix label smoothing with sigmoid")
                w = torch.softmax(torch.sigmoid(self.smoothing[y]), dim=1)
            else:
                w = torch.softmax(self.smoothing[y], dim=1)
            w = torch.softmax(self.smoothing[y], dim=1)
        #return (w*nls).sum(dim=1).mean(dim=0)
        return torch.nn.functional.kl_div(z.log_softmax(dim=1), w, reduction="batchmean")