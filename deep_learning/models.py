import torch
import torch.nn as nn
from utils import get_pulling_func
from copy import deepcopy


class InvarianceModel(nn.Module):
    def __init__(self, feature_extractor, pulling_op):
        super(InvarianceModel, self).__init__()
        self.feature_extractor = deepcopy(feature_extractor)
        self.pulling_op = get_pulling_func(pulling_op)

    def forward(self, x):
        mask = x[:, :, 0] != 0
        x = self.feature_extractor(x)
        x = self.pulling_op(x, mask, dim=1)
        return x


class DeepProp(nn.Module):
    def __init__(self, model_args, n_experiments=1):
        super(DeepProp, self).__init__()
        # self.added_features = 0
        # if model_args['model']['experiment_feature']:
        #     self.added_features += 1
        # if model_args['model']['pair_degree_feature']:
        #     self.added_features += 2
        self.pair_degree_feature = model_args['pair_degree_feature']
        self.n_experiments = n_experiments
        self.experiment_embedding_size = model_args['exp_emb_size']

        if self.experiment_embedding_size:
            self.experiment_embedding = torch.nn.Embedding(self.n_experiments, self.experiment_embedding_size)
            self.register_buffer('experiment_vector', torch.arange(n_experiments,dtype=torch.long))
        else:
            self.experiment_embedding = None

        feature_extractor = self.init_feature_extractor(model_args['feature_extractor_layers'])
        classifier = self.init_classifier(model_args['feature_extractor_layers'][-1],
                                          model_args['classifier_layers'],
                                          model_args['classifier_dropout']
                                          )

        if model_args['share_source_terminal_weights']:
            self.source_model = InvarianceModel(feature_extractor, model_args['pulling_func'])
            self.terminal_model = self.source_model
        else:
            self.source_model = InvarianceModel(feature_extractor, model_args['pulling_func'])
            self.terminal_model = InvarianceModel(feature_extractor, model_args['pulling_func'])
        self.classifier = classifier

    def init_feature_extractor(self, f_layers_size, dropout_rate=0):
        f_layers_size = [2] + f_layers_size
        feature_extractor_layers = []
        for idx in range(len(f_layers_size))[:-1]:
            feature_extractor_layers.append(nn.Linear(f_layers_size[idx], f_layers_size[idx + 1]))
            feature_extractor_layers.append(nn.ReLU(inplace=True))
            if dropout_rate:
                feature_extractor_layers.append(torch.nn.Dropout(dropout_rate))
        return nn.Sequential(*feature_extractor_layers[:-2 if dropout_rate else -1])

    def init_classifier(self, last_feature_dim, c_layers_size, dropout_rate=None):
        classifier_layers = []
        num_added_features = self.experiment_embedding_size + (2 if self.pair_degree_feature else 0)
        classifier_layers.append(
            nn.Linear((2 * last_feature_dim) + num_added_features, c_layers_size[0]))
        classifier_layers.append(nn.ReLU(inplace=True))

        for idx in range(len(c_layers_size))[:-1]:
            classifier_layers.append(nn.Linear(c_layers_size[idx], c_layers_size[idx + 1]))
            classifier_layers.append(nn.ReLU(inplace=True))
            if dropout_rate:
                classifier_layers.append(torch.nn.Dropout(dropout_rate))
        classifier_layers.append(nn.Linear(c_layers_size[-1], 1))
        return nn.Sequential(*classifier_layers)

    def forward(self, s, t, pair_degree):
        x_s = self.source_model(s)
        x_t = self.terminal_model(t)
        combined = torch.cat([x_s, x_t], 1)
        if self.experiment_embedding_size:
            experiment_vector = self.experiment_vector.repeat(int(x_s.shape[0]/self.n_experiments))
            experiment_embeddings = self.experiment_embedding(experiment_vector)
            combined = torch.cat([combined, experiment_embeddings], 1)
        if self.pair_degree_feature:
            pair_degree_feature = pair_degree.repeat_interleave(repeats=self.n_experiments, dim=0)
            combined = torch.cat([combined, pair_degree_feature], dim=1)

        logits = self.classifier(combined)
        return logits


class DeepPropClassifier(nn.Module):
    def __init__(self, deep_prop_model):
        super(DeepPropClassifier, self).__init__()
        self.n_experiments = deep_prop_model.n_experiments
        self.base_model = deep_prop_model
        self.classifier = nn.Sequential(nn.Linear(self.n_experiments, 2))

    def forward(self, s, t, pair_degree):
        flatten_s = torch.reshape(s, (-1, s.shape[2], s.shape[3]))
        flatten_t = torch.reshape(t, (-1, t.shape[2], t.shape[3]))
        combined = self.base_model(flatten_s, flatten_t, pair_degree)
        combined_2 = torch.reshape(combined, (-1, self.n_experiments))
        out = self.classifier(combined_2)
        pred = torch.argmax(out, dim=1)
        return out, pred, combined
