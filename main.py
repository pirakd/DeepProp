import pandas
from os import path
from deep_learning.data_loaders import ClassifierDataset, train_test_split
from deep_learning.models import DeepPropClassifier, DeepProp
from deep_learning.trainer import ClassifierTrainer
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from utils import read_data_single_type, generate_feature_columns_DSS, normalize_features
import torch
input_file = path.join('DrugsInfo', 'input')
NETWORK_FILENAME = path.join(input_file, "H_sapiens.net")
DIRECTED_INTERACTIONS_FILENAME = path.join(input_file, "consensus.net")
SOURCES_FILENAME = path.join(input_file, "drug_targets.txt")
TERMINALS_FILENAME = path.join(input_file, "drug_expressions.txt")
torch.set_default_dtype(torch.float64)

PROPAGATE_ALPHA = 0.8
PROPAGATE_EPSILON = 1e-6
PROPAGATE_ITERATIONS = 200
ORIENTATION_EPSILON = 0.01
n_epochs = 200
n_experiments = 20

network, directed_interactions, sources, terminals = read_data_single_type(NETWORK_FILENAME,
                                                                           DIRECTED_INTERACTIONS_FILENAME,
                                                                           SOURCES_FILENAME, TERMINALS_FILENAME, None)
merged_network = pandas.concat([directed_interactions, network.drop(directed_interactions.index & network.index)])
directed_interactions_pairs_list = tuple(directed_interactions.index)
pairs_to_index = {pair: p for p, pair in enumerate(merged_network.index)}

directed_interaction_set = set(directed_interactions_pairs_list)
labeled_pairs_to_index = {pair: idx for pair, idx in pairs_to_index.items() if pair in directed_interaction_set}

indexes_to_keep = list(labeled_pairs_to_index.values())
source_features, terminal_features = generate_feature_columns_DSS(merged_network, sources, terminals, indexes_to_keep,
                                                                  PROPAGATE_ALPHA, PROPAGATE_ITERATIONS,
                                                                  PROPAGATE_EPSILON,    n_experiments)
source_features, terminal_features = normalize_features(source_features, terminal_features)


train_indexes, test_indexes = train_test_split(len(indexes_to_keep), 0.8)
train_source_features = [x[train_indexes] for x in source_features]
test_source_features = [x[test_indexes] for x in source_features]
train_terminal_features = [x[train_indexes] for x in terminal_features]
test_terminal_features = [x[test_indexes] for x in terminal_features]
train_dataset = ClassifierDataset(train_source_features, train_terminal_features)
test_dataset = ClassifierDataset(test_source_features, test_terminal_features)
train_loader = DataLoader(train_dataset, batch_size=15, shuffle=True, pin_memory=True)
eval_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, pin_memory=True)
def protected_max(x, dim):
    return torch.max(x, dim=dim)[0]


feature_extractor = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16))
deep_prop_classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 1),)

pulling_op = torch.sum

classifier = nn.Sequential(nn.Linear(n_experiments, 2))

deep_prop_model = DeepProp(feature_extractor, pulling_op, deep_prop_classifier)
model = DeepPropClassifier(deep_prop_model, classifier, n_experiments)

optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)


trainer = ClassifierTrainer(n_epochs, criteria=nn.CrossEntropyLoss(), intermediate_criteria=nn.BCELoss(), optimizer=optimizer,
                  eval_metric=None, eval_interval=10, device='cpu')

trainer.train(train_loader=train_loader, eval_loader=eval_loader, model=model)
