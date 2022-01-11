import pandas
from os import path
from deep_learning.data_loaders import ClassifierDataset, train_test_split
from deep_learning.models import DeepPropClassifier, DeepProp
from deep_learning.trainer import ClassifierTrainer
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from utils import read_data, generate_feature_columns, normalize_features, get_pulling_func
import torch

input_file = path.join('DrugsInfo', 'input')
NETWORK_FILENAME = path.join(input_file, "H_sapiens.net")
DIRECTED_INTERACTIONS_FILENAME = path.join(input_file, "consensus.net")
SOURCES_FILENAME = path.join(input_file, "drug_targets.txt")
TERMINALS_FILENAME = path.join(input_file, "drug_expressions.txt")
torch.set_default_dtype(torch.float64)

args = {
    'data':
        {'n_experiments': 10,
         'train_test_split': 0.8,
         'dataset_type': 'balanced_kpi'},
    'propagation':
        {'alpha': 0.8,
         'eps': 1e-6,
         'n_iterations': 100},
    'model':
        {'feature_extractor_layers': [64, 32, 16],
         'classifier_layers': [32, 16, 8],
         'pulling_func': 'mean',
         'exp_emb_size': 4},
    'train':
        {'intermediate_loss_weight': 0.95,
         'train_test_ratio': 0.8,
         'train_batch_size': 16,
         'test_batch_size': 32,
         'n_epochs': 200,
         'eval_interval': 5,
         'learning_rate': 1e-3}}

# data read
network, directed_interactions, sources, terminals =\
    read_data(NETWORK_FILENAME, DIRECTED_INTERACTIONS_FILENAME, SOURCES_FILENAME, TERMINALS_FILENAME,
                                                               args['data']['dataset_type'])
merged_network = pandas.concat([directed_interactions, network.drop(directed_interactions.index & network.index)])
directed_interactions_pairs_list = tuple(directed_interactions.index)
pairs_to_index = {pair: p for p, pair in enumerate(merged_network.index)}

directed_interaction_set = set(directed_interactions_pairs_list)
labeled_pairs_to_index = {pair: idx for pair, idx in pairs_to_index.items() if pair in directed_interaction_set}

indexes_to_keep = list(labeled_pairs_to_index.values())

# feature generation
source_features, terminal_features = generate_feature_columns(merged_network, sources, terminals, indexes_to_keep,
                                                              args['propagation']['alpha'], args['propagation']['n_iterations'],
                                                              args['propagation']['eps'], args['data']['n_experiments'])

#pro processing
source_features, terminal_features = normalize_features(source_features, terminal_features)

train_indexes, test_indexes = train_test_split(len(indexes_to_keep), args['train']['train_test_ratio'])

train_source_features = [x[train_indexes] for x in source_features]
test_source_features = [x[test_indexes] for x in source_features]
train_terminal_features = [x[train_indexes] for x in terminal_features]
test_terminal_features = [x[test_indexes] for x in terminal_features]

train_dataset = ClassifierDataset(train_source_features, train_terminal_features)
test_dataset = ClassifierDataset(test_source_features, test_terminal_features)
train_loader = DataLoader(train_dataset, batch_size=args['train']['train_batch_size'], shuffle=True, pin_memory=True)
eval_loader = DataLoader(test_dataset, batch_size=args['train']['test_batch_size'], shuffle=False, pin_memory=True)

pulling_func = get_pulling_func(args['model']['pulling_func'])

deep_prop_model = DeepProp(args['model']['feature_extractor_layers'], pulling_func, args['model']['classifier_layers'], args['data']['n_experiments'],
                           args['model']['exp_emb_size'])

model = DeepPropClassifier(deep_prop_model, args['data']['n_experiments'])
optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

trainer = ClassifierTrainer(args['train']['n_epochs'], criteria=nn.CrossEntropyLoss(), intermediate_criteria=nn.BCELoss(),
                            intermediate_loss_weight=args['train']['intermediate_loss_weight'],
                            optimizer=optimizer, eval_metric=None, eval_interval=args['train']['eval_interval'], device='cpu')

trainer.train(train_loader=train_loader, eval_loader=eval_loader, model=model)
