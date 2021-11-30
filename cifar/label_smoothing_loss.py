import torch
import torch.nn as nn

class LabelSmoothingLoss(nn.Module):
    def __init__(self, prediction_size, epsilon=None):
        super(LabelSmoothingLoss, self).__init__()
        self.prediction_size = prediction_size
        if epsilon == None:
            #self.smoothing = nn.Parameter(-torch.rand((1)))
            self.smoothing = nn.Parameter(-torch.rand((prediction_size)))
        else:
            self.smoothing = nn.Parameter(epsilon.clone().detach())

    def forward(self, prediction, target):
        neglog_softmaxPrediction = -prediction.log_softmax(dim=1)
        reparameterized_smoothing = torch.sigmoid(self.smoothing)
        weights = torch.ones_like(prediction)*reparameterized_smoothing/(prediction.shape[1]-1)
        smoothedLabels = torch.scatter(weights, 1, target.unsqueeze(1), torch.ones((prediction.shape[0],1), device="cuda")-reparameterized_smoothing)
        return torch.mean(torch.sum(smoothedLabels*neglog_softmaxPrediction, dim=1))