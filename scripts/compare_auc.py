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
    load_pickle, train_test_split, get_normalization_constants, get_loss_function, get_time, gen_propagation_scores
from D2D import eval_D2D, eval_D2D_2, generate_D2D_features_from_propagation_scores
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
from presets import experiments_20, experiments_50, experiments_all
import json
import argparse
from scripts.scripts_utils import sources_filenmae_dict, terminals_filenmae_dict
torch.set_default_dtype(torch.float32)


def run(sys_args):
    n_folds = sys_args.n_folds
    root_path = get_root_path()
    script_name = path.basename(__file__).split('.')[0]
    output_folder = 'output'
    output_file_path = path.join(get_root_path(), output_folder, path.basename(__file__).split('.')[0], get_time())
    makedirs(output_file_path, exist_ok=True)
    n_experiments = sys_args.n_experiments
    if  n_experiments == 0:
        args = experiments_all
    elif n_experiments <= 30:
        args = experiments_20
    else:
        args = experiments_50

    device = torch.device("cuda:{}".format(sys_args.device) if torch.cuda.is_available() else "cpu")
    args['data']['load_prop_scores'] = sys_args.load_prop_scores
    args['data']['save_prop_scores'] = sys_args.save_prop_scores
    args['data']['prop_scores_filename'] = sys_args.prop_scores_filename
    args['train']['train_val_test_split'] = sys_args.train_val_test_split
    args['data']['directed_interactions_filename'] = sys_args.directed_interactions_filename
    args['data']['sources_filename'] = sources_filenmae_dict[sys_args.experiments_type]
    args['data']['terminals_filename'] = terminals_filenmae_dict[sys_args.experiments_type]
    args['data']['n_experiments']  = n_experiments

    # data read and filtering
    rng = np.random.RandomState(args['data']['random_seed'])

    network, directed_interactions, sources, terminals, id_to_degree =\
        read_data(args['data']['network_filename'], args['data']['directed_interactions_filename'],
                  args['data']['sources_filename'], args['data']['terminals_filename'],
                  args['data']['n_experiments'], args['data']['max_set_size'], rng)
    n_experiments = len(sources)

    # merged_network = pandas.concat([directed_interactions, network.drop(directed_interactions.index & network.index)])
    directed_interactions_pairs_list = np.array(directed_interactions.index)
    directed_interactions_source_type = np.array(directed_interactions.source)
    genes_ids_to_keep = sorted(list(set([x for pair in directed_interactions_pairs_list for x in pair])))

    propagation_scores, row_id_to_idx, col_id_to_idx, normalization_constants_dict = \
        gen_propagation_scores(args, network, sources, terminals, genes_ids_to_keep, directed_interactions_pairs_list)

    deep_probs_list, deep_labels_list, d2d_probs_list_2, d2d_labels_list_2, d2d_probs_list, d2d_labels_list = [],[],[],[],[],[]
    for fold in range(n_folds):
        train_indexes, val_indexes, test_indexes = train_test_split(len(directed_interactions_pairs_list),
                                                                    args['train']['train_val_test_split'],
                                                                    random_state=rng)    # feature generation
        d2d_train_indexes = np.concatenate([train_indexes, val_indexes])

        # d2d evaluation
        sources_indexes = [[row_id_to_idx[gene_id] for gene_id in gene_set] for gene_set in sources.values()]
        terminals_indexes = [[row_id_to_idx[gene_id] for gene_id in gene_set] for gene_set in terminals.values()]
        pairs_indexes = [(col_id_to_idx[pair[0]], col_id_to_idx[pair[1]]) for pair in directed_interactions_pairs_list]
        features, deconstructed_features = generate_D2D_features_from_propagation_scores(propagation_scores, pairs_indexes,
                                                                                         sources_indexes, terminals_indexes)
        d2d_probs, d2d_labels = eval_D2D(features[d2d_train_indexes], features[test_indexes])
        d2d_probs_2, d2d_labels_2 = eval_D2D_2(deconstructed_features[d2d_train_indexes], deconstructed_features[test_indexes])

        train_indexes, val_indexes, test_indexes = train_test_split(len(directed_interactions_pairs_list),
                                                                    args['train']['train_val_test_split'],
                                                                    random_state=rng)
        train_dataset = LightDataset(row_id_to_idx, col_id_to_idx, propagation_scores,
                                     directed_interactions_pairs_list[train_indexes], sources, terminals,
                                     args['data']['normalization_method'],
                                     normalization_constants_dict, directed_interactions_source_type[train_indexes],
                                     id_to_degree)
        train_loader = DataLoader(train_dataset, batch_size=args['train']['train_batch_size'], shuffle=True,
                                  pin_memory=False)
        val_dataset = LightDataset(row_id_to_idx, col_id_to_idx, propagation_scores,
                                   directed_interactions_pairs_list[val_indexes], sources, terminals,
                                   args['data']['normalization_method'],
                                   normalization_constants_dict, directed_interactions_source_type[val_indexes],
                                   id_to_degree)
        val_loader = DataLoader(val_dataset, batch_size=args['train']['test_batch_size'], shuffle=False,
                                pin_memory=False)
        test_dataset = LightDataset(row_id_to_idx, col_id_to_idx, propagation_scores,
                                    directed_interactions_pairs_list[test_indexes], sources, terminals,
                                    args['data']['normalization_method'],
                                    normalization_constants_dict, directed_interactions_source_type[test_indexes],
                                    id_to_degree)
        test_loader = DataLoader(test_dataset, batch_size=args['train']['test_batch_size'], shuffle=False,
                                 pin_memory=False, )

        # build models
        deep_prop_model = DeepProp(args['model'], n_experiments)

        model = DeepPropClassifier(deep_prop_model)
        model.to(device=device)


        # train
        intermediate_loss_type = get_loss_function(args['train']['intermediate_loss_type'],
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

if __name__ == '__main__':
    n_folds = 3
    input_type = 'drug'
    load_prop = False
    save_prop = False
    n_exp = 2
    split = [0.66, 0.14, 0.2]
    interaction_type = ['KPI']
    device = 'cpu'
    prop_scores_filename = 'drug_KPI'

    parser = argparse.ArgumentParser()
    parser.add_argument('-ex', '--ex_type', dest='experiments_type', type=str,
                        help='name of experiment type(drug, colon, etc.)', default=input_type)
    parser.add_argument('-n,', '--n_exp', dest='n_experiments', type=int,
                        help='num of experiments used (0 for all)', default=n_exp)
    parser.add_argument('-s', '--save_prop', dest='save_prop_scores', action='store_true', default=False,
                        help='Whether to save propagation scores')
    parser.add_argument('-l', '--load_prop', dest='load_prop_scores', action='store_true', default=False,
                        help='Whether to load prop scores')
    parser.add_argument('-sp', '--split', dest='train_val_test_split', nargs=3, help='[train, val, test] sums to 1',
                        default=split, type=float)
    parser.add_argument('-d', '--device', type=str, help='cpu or gpu number', default=device)
    parser.add_argument('-in', '--inter_file', dest='directed_interactions_filename', type=str, help='KPI/STKE',
                        default=interaction_type)
    parser.add_argument('-p', '--prop_file', dest='prop_scores_filename', type=str,
                        help='Name of prop score file(save/load)', default=prop_scores_filename)

    parser.add_argument('-f', '--n_folds', dest='prop_scores_filename', type=str,
                        help='Name of prop score file(save/load)', default=n_folds)
    args = parser.parse_args()
    # args.load_prop_scores =  True
    # args.save_prop_scores =  True
    run(args)
