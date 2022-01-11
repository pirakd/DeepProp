from torch import nn
from torch.autograd import Variable
from torch.functional import F


def MaskedMean(x, mask, dim):
    mask = mask.int().unsqueeze(-1)
    return (x * mask).sum(dim) / mask.sum(dim)


def MaskedSum(x, mask, dim):
    mask = mask.int().unsqueeze(-1)
    return (x * mask).sum(dim)


class FocalLoss(nn.Module):
    def __init__(self, gamma=0.8, reduction='sum'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, x, y):
        y = y.view(-1, 1)

        logpt = F.log_softmax(x, dim=1)
        logpt = logpt.gather(1, y)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())
        loss = -1 * (1-pt)**self.gamma * logpt

        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()

        else:
            assert 0, 'no such reduction for FocalLoss'