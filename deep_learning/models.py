import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# class DSSLinearLayer(nn.Module):
#     def __init__(self, input_dim, output_dim, op=torch.sum):
#         super(DSSLinearLayer, self).__init__()
#         self.fc = nn.Linear(input_dim, output_dim)
#         self.bn = nn.BatchNorm1d(output_dim)
#         self.shared_fc = nn.Linear(input_dim, output_dim)
#         self.shared_bn = nn.BatchNorm1d(output_dim)
#         self.op = op
#
#     def forward(self, x):
#         x_i = self.fc(x)
#         bs, n, d = x_i.shape
#         x_i = self.bn(x_i.view(bs * n, d)).view(bs, n, d)
#         # x_s = self.shared_fc(self.op(x, 1))
#         # x_s = self.shared_bn(x_s)
#         # o = x_s.unsqueeze(1).repeat([1, n, 1]) + x_i
#         return x_i


class InvarianceModel(nn.Module):
    def __init__(self, feature_extractor, pulling_op, drop_rate=None):
        super(InvarianceModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.pulling_op = pulling_op

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.pulling_op(x, dim=1)
        return x



class DeepProp(nn.Module):
    def __init__(self, feature_extractor, pulling_op, classifier):
        super(DeepProp, self).__init__()
        self.pulling_op = pulling_op
        self.source_model = InvarianceModel(feature_extractor, pulling_op)
        self.terminal_model = InvarianceModel(feature_extractor, pulling_op)
        self.classifier = classifier

    def forward(self, s, t):
        x_s = self.source_model(s)
        x_t = self.terminal_model(t)
        combined = torch.cat([x_s, x_t], 1)
        logits = self.classifier(combined)
        pred = torch.argmax(logits, dim=1)
        return logits, pred


class DeepPropClassifier(nn.Module):
    def __init__(self, deep_prop_model, classifier, n_experiments):
        super(DeepPropClassifier, self).__init__()
        self.n_experiments = n_experiments
        self.base_model = deep_prop_model
        # self.classifier = nn.Sequential(nn.Linear(n_experiments, 2))
        self.classifier = classifier

    def forward(self, s, t):
        flatten_s = torch.reshape(s, (-1, s.shape[2], s.shape[3]))
        flatten_t = torch.reshape(t, (-1, t.shape[2], t.shape[3]))

        combined = self.base_model(flatten_s, flatten_t)[0]
        combined_2 = torch.reshape(combined, (-1, self.n_experiments))
        out = self.classifier(combined_2)
        pred = torch.argmax(out, dim=1)
        return out, pred, combined


class FocalLoss(nn.Module):
    def __init__(self, gamma=0.7, reduction='sum'):
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
            assert 0, 'no such reduction in FocalLoss'