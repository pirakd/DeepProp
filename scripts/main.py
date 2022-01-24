from os import path
from deep_learning.data_loaders import ClassifierDataset
from deep_learning.models import DeepPropClassifier, DeepProp
from deep_learning.trainer import ClassifierTrainer
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from utils import read_data, generate_feature_columns, normalize_features, get_root_path, train_test_split
import torch
from sklearn.metrics import precision_recall_curve, auc

root_path = get_root_path()
input_file = path.join(root_path, 'input')
NETWORK_FILENAME = path.join(input_file, 'networks', "H_sapiens.net")
DIRECTED_INTERACTIONS_FILENAME = path.join(input_file, 'directed_interactions', "KPI_dataset")
SOURCES_FILENAME = path.join(input_file, 'priors', "drug_targets.txt")
TERMINALS_FILENAME = path.join(input_file, 'priors', "drug_expressions.txt")
torch.set_default_dtype(torch.float64)

args = {
    'data':
        {'n_experiments': 20,
         'train_test_split': 0.8},
    'propagation':
        {'alpha': 0.8,
         'eps': 1e-6,
         'n_iterations': 200},
    'model':
        {'feature_extractor_layers': [64, 32, 16],
         'classifier_layers': [128,64],
         'pulling_func': 'mean',
         'exp_emb_size': 8},
    'train':
        {'intermediate_loss_weight': 0,
         'train_val_test_split' : [0.66, 0.14, 0.2], # sum([train, val, test])=1
         'train_batch_size': 32,
         'test_batch_size': 32,
         'n_epochs': 100,
         'eval_interval': 5,
         'learning_rate': 1e-3,
         'n_evals_no_improvement': 10}}


# data read
network, directed_interactions, sources, terminals =\
    read_data(NETWORK_FILENAME, DIRECTED_INTERACTIONS_FILENAME, SOURCES_FILENAME, TERMINALS_FILENAME)
# merged_network = pandas.concat([directed_interactions, network.drop(directed_interactions.index & network.index)])
directed_interactions_pairs_list = tuple(directed_interactions.index)
pairs_to_index = {pair: p for p, pair in enumerate(network.index)}

directed_interaction_set = set(directed_interactions_pairs_list)
labeled_pairs_to_index = {pair: idx for pair, idx in pairs_to_index.items() if pair in directed_interaction_set}

indexes_to_keep = list(labeled_pairs_to_index.values())

# feature generation
source_features, terminal_features = generate_feature_columns(network, sources, terminals, indexes_to_keep,
                                                              args['propagation']['alpha'], args['propagation']['n_iterations'],
                                                              args['propagation']['eps'], args['data']['n_experiments'])
train_indexes, val_indexes, test_indexes = train_test_split(len(indexes_to_keep), args['train']['train_val_test_split'])


#pro processing
source_features, terminal_features = normalize_features(source_features, terminal_features)

test_source_features = [x[test_indexes] for x in source_features]
test_terminal_features = [x[test_indexes] for x in terminal_features]
test_dataset = ClassifierDataset(test_source_features, test_terminal_features)
test_loader = DataLoader(test_dataset, batch_size=args['train']['test_batch_size'], shuffle=False, pin_memory=True)
train_source_features = [x[train_indexes] for x in source_features]
train_terminal_features = [x[train_indexes] for x in terminal_features]
train_dataset = ClassifierDataset(train_source_features, train_terminal_features)
train_loader = DataLoader(train_dataset, batch_size=args['train']['train_batch_size'], shuffle=True, pin_memory=True)
val_source_features = [x[val_indexes] for x in source_features]
val_terminal_features = [x[val_indexes] for x in terminal_features]
val_dataset = ClassifierDataset(val_source_features, val_terminal_features)
val_loader = DataLoader(val_dataset, batch_size=args['train']['test_batch_size'], shuffle=False, pin_memory=True)


deep_prop_model = DeepProp(args['model']['feature_extractor_layers'], args['model']['pulling_func'],
                           args['model']['classifier_layers'], args['data']['n_experiments'],
                           args['model']['exp_emb_size'])
model = DeepPropClassifier(deep_prop_model, args['data']['n_experiments'])


optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args['train']['learning_rate'])

trainer = ClassifierTrainer(args['train']['n_epochs'], criteria=nn.CrossEntropyLoss(), intermediate_criteria=nn.BCELoss(),
                            intermediate_loss_weight=args['train']['intermediate_loss_weight'],
                            optimizer=optimizer, eval_metric=None, eval_interval=args['train']['eval_interval'], device='cpu')

train_stats, best_model = trainer.train(train_loader=train_loader, eval_loader=val_loader, model=model,
                                        max_evals_no_improvement=args['train']['n_evals_no_improvement'])
deep_probs, deep_labels = trainer.eval(best_model, test_loader, in_train=False)
deep_precision, deep_recall, deep_thresholds = precision_recall_curve(deep_labels, deep_probs)
deep_auc = auc(deep_recall, deep_precision)

print('Test AUC is {:.2f}'.format(deep_auc))


