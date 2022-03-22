from os import path
import sys
sys.path.append(path.dirname(path.dirname(path.realpath(__file__))))
from os import path, makedirs
from deep_learning.data_loaders import ClassifierDataset, LightDataset
from deep_learning.models import DeepPropClassifier, DeepProp
from deep_learning.trainer import ClassifierTrainer
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch
from utils import read_data, generate_raw_propagation_scores, log_results, get_root_path, save_propagation_score,\
    load_pickle, train_test_split, get_normalization_constants, get_loss_function, get_time
from D2D import eval_D2D, eval_D2D_2, generate_D2D_features_from_propagation_scores
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
from presets import experiments_20, experiments_50, experiments_all
import json
sources_filenmae_dict = {'drug': 'targets_drug',
                         'AML': 'mutations_AML',
                         'colon': 'mutations_colon',
                         'ovary': 'mutations_ovary',
                         'breast': 'mutations_breast'}
terminals_filenmae_dict = {'drug': 'expressions_drug',
                           'AML': 'gene_expression_AML',
                           'colon': 'gene_expression_colon',
                           'ovary': 'gene_expression_ovary',
                           'breast': 'gene_expression_breast'}

script_name = path.basename(__file__).split('.')[0]
root_path = get_root_path()

torch.set_default_dtype(torch.float32)
output_folder = path.join(root_path, 'output', script_name, get_time())
makedirs(output_folder)
device = 'cpu'
n_folds = 3
n_experiments = 1
input_name = 'drug'

cmd_args = [arg for arg in sys.argv[1:]]
print(cmd_args)
if len(cmd_args) == 3:
    input_name = cmd_args[0]
    n_experiments = cmd_args[2] if cmd_args[2] == 'all' else int(cmd_args[2])
    device = torch.device("cuda:{}".format(int(cmd_args[1])) if torch.cuda.is_available() else "cpu")
print('{} {} {}'.format(input_name, device, n_experiments))

if n_experiments == 'all' or n_experiments >= 100:
    args = experiments_all
elif n_experiments <= 30:
    args = experiments_20
else:
    args = experiments_50

args['data']['sources_filename'] = sources_filenmae_dict[input_name]
args['data']['terminals_filename'] = terminals_filenmae_dict[input_name]
args['data']['n_experiments'] = n_experiments
args['data']['save_prop_scores'] = input_name


# data read and filtering
rng = np.random.RandomState(args['data']['random_seed'])

network, directed_interactions, sources, terminals =\
    read_data(args['data']['network_filename'], args['data']['directed_interactions_filename'],
              args['data']['sources_filename'], args['data']['terminals_filename'],
              args['data']['n_experiments'], args['data']['max_set_size'], rng)
n_experiments = len(sources)

# merged_network = pandas.concat([directed_interactions, network.drop(directed_interactions.index & network.index)])
directed_interactions_pairs_list = np.array(directed_interactions.index)
genes_ids_to_keep = sorted(list(set([x for pair in directed_interactions_pairs_list for x in pair])))

if args['data']['load_prop_scores']:
    scores_file_path = path.join(root_path, 'input', 'propagation_scores', args['data']['prop_scores_filename'])
    scores_dict = load_pickle(scores_file_path)
    propagation_scores = scores_dict['propagation_scores']
    row_id_to_idx, col_id_to_idx = scores_dict['row_id_to_idx'], scores_dict['col_id_to_idx']
    normalization_constants_dict = scores_dict['normalization_constants']
    sources_indexes = [[row_id_to_idx[gene_id] for gene_id in gene_set] for gene_set in sources.values()]
    terminals_indexes = [[row_id_to_idx[gene_id] for gene_id in gene_set] for gene_set in terminals.values()]
    pairs_indexes = [(col_id_to_idx[pair[0]], col_id_to_idx[pair[1]]) for pair in directed_interactions_pairs_list]
    assert scores_dict['data_args']['random_seed'] == args['data']['random_seed'],\
        'random seed of loaded data does not much current one'
else:
    propagation_scores, row_id_to_idx, col_id_to_idx =\
        generate_raw_propagation_scores(network, sources, terminals, genes_ids_to_keep, args['propagation']['alpha'],
                                        args['propagation']['n_iterations'], args['propagation']['eps'])
    sources_indexes = [[row_id_to_idx[gene_id] for gene_id in gene_set] for gene_set in sources.values()]
    terminals_indexes = [[row_id_to_idx[gene_id] for gene_id in gene_set] for gene_set in terminals.values()]
    pairs_indexes = [(col_id_to_idx[pair[0]], col_id_to_idx[pair[1]]) for pair in directed_interactions_pairs_list]
    normalization_constants_dict = get_normalization_constants(pairs_indexes, sources_indexes, terminals_indexes,
                                                               propagation_scores)
    if args['data']['save_prop_scores']:
        save_propagation_score(propagation_scores, normalization_constants_dict, row_id_to_idx, col_id_to_idx,
                               args['propagation'], args['data'], 'balanced_kpi_prop_scores')

deep_probs_list, deep_labels_list, d2d_probs_list_2, d2d_labels_list_2, d2d_probs_list, d2d_labels_list = [],[],[],[],[],[]
for fold in range(n_folds):
    train_indexes, val_indexes, test_indexes = train_test_split(len(directed_interactions_pairs_list),
                                                                args['train']['train_val_test_split'],
                                                                random_state=rng)    # feature generation
    d2d_train_indexes = np.concatenate([train_indexes, val_indexes])

    # d2d evaluation

    features, deconstructed_features = generate_D2D_features_from_propagation_scores(propagation_scores, pairs_indexes,
                                                                                     sources_indexes, terminals_indexes)
    d2d_probs, d2d_labels = eval_D2D(features[train_indexes], features[test_indexes])
    d2d_probs_2, d2d_labels_2 = eval_D2D_2(deconstructed_features[train_indexes], deconstructed_features[test_indexes])

    train_dataset = LightDataset(row_id_to_idx, col_id_to_idx, propagation_scores,
                                 directed_interactions_pairs_list[train_indexes], sources, terminals,
                                 normalization_constants_dict)
    val_dataset = LightDataset(row_id_to_idx, col_id_to_idx, propagation_scores,
                               directed_interactions_pairs_list[val_indexes], sources, terminals,
                               normalization_constants_dict)
    test_dataset = LightDataset(row_id_to_idx, col_id_to_idx, propagation_scores,
                                directed_interactions_pairs_list[test_indexes], sources, terminals,
                                normalization_constants_dict)

    train_loader = DataLoader(train_dataset, batch_size=args['train']['train_batch_size'], shuffle=True,
                              pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=args['train']['test_batch_size'], shuffle=False, pin_memory=False, )
    test_loader = DataLoader(test_dataset, batch_size=args['train']['test_batch_size'], shuffle=False,
                             pin_memory=False, )

    # build models
    deep_prop_model = DeepProp(args['model']['feature_extractor_layers'], args['model']['pulling_func'],
                               args['model']['classifier_layers'], n_experiments,
                               args['model']['exp_emb_size'])
    model = DeepPropClassifier(deep_prop_model)
    model.to(device=device)


    # train
    intermediate_loss_type =  get_loss_function(args['train']['intermediate_loss_type'],
                                                                focal_gamma=args['train']['focal_gamma'])
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args['train']['learning_rate'])
    trainer = ClassifierTrainer(args['train']['n_epochs'], criteria=nn.CrossEntropyLoss(),
                                intermediate_criteria=intermediate_loss_type,
                                intermediate_loss_weight=args['train']['intermediate_loss_weight'],
                                optimizer=optimizer, eval_metric=None, eval_interval=args['train']['eval_interval'],
                                device=device)

    train_stats, best_model = trainer.train(train_loader=train_loader, eval_loader=val_loader, model=model,
                                            max_evals_no_improvement=args['train']['max_evals_no_imp'])

    deep_probs, deep_labels = trainer.eval(best_model, test_loader, output_probs=True)

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
plt.title('{}, {}, {} sources, {} folds'.format(args['data']['directed_interactions_filename'],
                                                args['data']['sources_filename'].split('_')[-1],
                                                n_experiments, n_folds))

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
