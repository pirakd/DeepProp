import torch
import torch.nn as nn

class InvarianceModel(nn.Module):
    def __init__(self, feature_extractor, pulling_op, drop_rate=None):
        super(InvarianceModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.pulling_op = pulling_op

    def forward(self, x):
        mask = x[:,:,0] != 0
        x = self.feature_extractor(x)
        x = self.pulling_op(x, mask, dim=1)
        return x

class DeepProp(nn.Module):
    def __init__(self, feature_extractor, pulling_op, classifier, n_experiments=1, experiment_feature_size=None):
        super(DeepProp, self).__init__()
        self.pulling_op = pulling_op
        self.source_model = InvarianceModel(feature_extractor, pulling_op)
        self.terminal_model = InvarianceModel(feature_extractor, pulling_op)
        self.classifier = classifier
        self.n_experiments = n_experiments
        self.experiment_feature_size = experiment_feature_size
        self.experiment_embedding = torch.nn.Embedding(self.n_experiments, self.experiment_feature_size)

    def forward(self, s, t):
        x_s = self.source_model(s)
        x_t = self.terminal_model(t)
        combined = torch.cat([x_s, x_t], 1)
        if self.experiment_feature_size:
            experiment_vector = torch.arange(0, self.n_experiments, dtype=torch.int).repeat(int(x_s.shape[0]/self.n_experiments))
            experiment_embeddings = self.experiment_embedding(experiment_vector)
            combined = torch.cat([combined, experiment_embeddings], 1)
        logits = self.classifier(combined)
        return logits


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

        combined = self.base_model(flatten_s, flatten_t)
        combined_2 = torch.reshape(combined, (-1, self.n_experiments))
        out = self.classifier(combined_2)
        pred = torch.argmax(out, dim=1)
        return out, pred, combined
