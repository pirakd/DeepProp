from os import path
import sys
sys.path.append(path.dirname(path.dirname(path.realpath(__file__))))
from os import path, makedirs
from deep_learning.data_loaders import ClassifierDataset
from deep_learning.models import DeepPropClassifier, DeepProp
from deep_learning.trainer import ClassifierTrainer
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from utils import read_data, generate_feature_columns, normalize_features, get_root_path, train_test_split, get_time
import torch
from D2D import generate_D2D_features, eval_D2D, generate_D2D_features_2, eval_D2D_2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
from presets import experiments_20, experiments_50
import json

script_name = path.basename(__file__).split('.')[0]
root_path = get_root_path()
input_file = path.join(root_path, 'input')
NETWORK_FILENAME = path.join(input_file, 'networks', "H_sapiens.net")
DIRECTED_INTERACTIONS_FILENAME = path.join(input_file, 'directed_interactions', "KPI_dataset")
SOURCES_FILENAME = path.join(input_file, 'priors', "drug_targets.txt")
TERMINALS_FILENAME = path.join(input_file, 'priors', "drug_expressions.txt")
torch.set_default_dtype(torch.float32)
random_state = np.random.RandomState(0)
output_folder = path.join(root_path, 'output', script_name, get_time())
makedirs(output_folder)
device = 'cpu'
n_folds = 3
n_experiments = 50

cmd_args = [int(arg) for arg in sys.argv[1:]]
if len(cmd_args) == 2:
    n_experiments = cmd_args[1]
    device = torch.device("cuda:{}".format(cmd_args[0]) if torch.cuda.is_available() else "cpu")


if n_experiments <= 30:
    args = experiments_20
else:
    args = experiments_50

args['data']['n_experiments'] = n_experiments


# data read and filtering
network, directed_interactions, sources, terminals =\
    read_data(NETWORK_FILENAME, DIRECTED_INTERACTIONS_FILENAME, SOURCES_FILENAME, TERMINALS_FILENAME)
# merged_network = pandas.concat([directed_interactions, network.drop(directed_interactions.index & network.index)])
directed_interactions_pairs_list = tuple(directed_interactions.index)
pairs_to_index = {pair: p for p, pair in enumerate(network.index)}
directed_interaction_set = set(directed_interactions_pairs_list)
labeled_pairs_to_index = {pair: idx for pair, idx in pairs_to_index.items() if pair in directed_interaction_set}
indexes_to_keep = list(labeled_pairs_to_index.values())

source_features, terminal_features = generate_feature_columns(network, sources, terminals, indexes_to_keep,
                                                              args['propagation']['alpha'], args['propagation']['n_iterations'],
                                                              args['propagation']['eps'], args['data']['n_experiments'])

deep_probs_list, deep_labels_list, d2d_probs_list_2, d2d_labels_list_2, d2d_probs_list, d2d_labels_list = [],[],[],[],[],[]
for fold in range(n_folds):
    train_indexes, val_indexes, test_indexes = train_test_split(len(indexes_to_keep), args['train']['train_val_test_split'], random_state)
    # feature generation
    d2d_train_indexes = np.concatenate([train_indexes, val_indexes])

    # d2d evaluation

    features = generate_D2D_features(source_features, terminal_features)
    train_features, test_features = features[d2d_train_indexes], features[test_indexes]
    d2d_probs, d2d_labels = eval_D2D(train_features, test_features)

    features_2 = generate_D2D_features_2(source_features, terminal_features)
    train_features_2, test_features_2 = features_2[d2d_train_indexes], features_2[test_indexes]
    d2d_probs_2, d2d_labels_2 = eval_D2D_2(train_features_2, test_features_2)


    #pro processing
    source_features, terminal_features = normalize_features(source_features, terminal_features)

    # building datasets
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

    # build models
    deep_prop_model = DeepProp(args['model']['feature_extractor_layers'], args['model']['pulling_func'],
                               args['model']['classifier_layers'], args['data']['n_experiments'],
                               args['model']['exp_emb_size'])
    model = DeepPropClassifier(deep_prop_model, args['data']['n_experiments'])

    # train
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args['train']['learning_rate'])
    trainer = ClassifierTrainer(args['train']['n_epochs'], criteria=nn.CrossEntropyLoss(), intermediate_criteria=nn.BCELoss(),
                                intermediate_loss_weight=args['train']['intermediate_loss_weight'],
                                optimizer=optimizer, eval_metric=None, eval_interval=args['train']['eval_interval'], device=device)

    train_stats, best_model = trainer.train(train_loader=train_loader, eval_loader=val_loader, model=model,
                                            max_evals_no_improvement=args['train']['n_evals_no_improvement'])

    deep_probs, deep_labels = trainer.eval(best_model, test_loader, in_train=False)

    deep_probs_list.append(deep_probs[:, 1])
    deep_labels_list.append(deep_labels)
    d2d_probs_list.append(d2d_probs[:, 1])
    d2d_labels_list.append(d2d_labels)
    d2d_probs_list_2.append(d2d_probs_2[:, 1])
    d2d_labels_list_2.append(d2d_labels_2)

deep_probs = np.hstack(deep_probs_list)
deep_labels = np.hstack(deep_labels_list)
d2d_probs = np.hstack(d2d_probs_list)
d2d_labels = np.hstack(d2d_labels_list)
d2d_probs_2 = np.hstack(d2d_probs_list_2)
d2d_labels_2 = np.hstack(d2d_labels_list_2)

deep_precision, deep_recall, deep_thresholds = precision_recall_curve(deep_labels, deep_probs)
deep_auc = auc(deep_recall, deep_precision)

d2d_precision, d2d_recall, d2d_thresholds = precision_recall_curve(d2d_labels, d2d_probs)
d2d_auc = auc(d2d_recall, d2d_precision)

d2d_precision_2, d2d_recall_2, d2d_thresholds_2 = precision_recall_curve(d2d_labels_2, d2d_probs_2)
d2d_auc_2 = auc(d2d_recall_2, d2d_precision_2)


plt.plot([0, 1], [0.5, 0.5], '--', color=(0.8, 0.8, 0.8), label='random')
plt.plot(deep_recall, deep_precision, 'o--', marker='o', label='DeepSetPropagation (%0.2f)' % deep_auc, lw=2, markersize=3)  # , markeredgecolor = 'dimgrey')
plt.plot(d2d_recall, d2d_precision, 'o--', marker='o', label='D2D (%0.2f)' % d2d_auc, lw=2, markersize=3)  # , markeredgecolor = 'dimgrey')
plt.plot(d2d_recall_2, d2d_precision_2, 'o--', marker='o', label='D2D deconstructed (%0.2f)' % d2d_auc_2, lw=2, markersize=3)  # , markeredgecolor = 'dimgrey')

plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc="lower right")
plt.title('KPI Balanced, {} sources, {} folds'.format(args['data']['n_experiments'], n_folds))

params = {'legend.fontsize': 8,
          'figure.figsize': (4.6, 2.9),
          'axes.labelsize': 8,
          'axes.titlesize': 8,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8}

plt.rcParams.update(params)
plt.savefig(path.join(output_folder, 'auc_curve'))

with open(path.join(output_folder, 'args'), 'w') as f:
    json.dump(args, f, indent=4, separators=(',', ': '))
