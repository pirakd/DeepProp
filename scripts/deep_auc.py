from os import path
import sys

sys.path.append(path.dirname(path.dirname(path.realpath(__file__))))
from os import path, makedirs
from deep_learning.data_loaders import LightDataset
from deep_learning.models import DeepPropClassifier, DeepProp
from deep_learning.trainer import ClassifierTrainer
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch
from utils import read_data, get_root_path, save_model, train_test_split, get_loss_function, get_time, \
    gen_propagation_scores, redirect_output
from D2D import eval_D2D, eval_D2D_2, generate_D2D_features_from_propagation_scores
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
from presets import experiments_20, experiments_50, experiments_0
import json
import argparse
from scripts.scripts_utils import sources_filenmae_dict, terminals_filenmae_dict
from collections import defaultdict

n_models_per_fold = 2


def run(sys_args):
    output_folder = 'output'
    output_file_path = path.join(get_root_path(), output_folder, path.basename(__file__).split('.')[0], get_time())
    makedirs(output_file_path, exist_ok=True)
    redirect_output(path.join(output_file_path, 'log'))

    n_experiments = sys_args.n_experiments
    if n_experiments == 0:
        args = experiments_0
    elif n_experiments <= 30:
        args = experiments_20
    else:
        args = experiments_50

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args['data']['load_prop_scores'] = sys_args.load_prop_scores
    args['data']['save_prop_scores'] = sys_args.save_prop_scores
    args['data']['prop_scores_filename'] = sys_args.prop_scores_filename
    args['train']['train_val_test_split'] = sys_args.train_val_test_split
    args['data']['directed_interactions_filename'] = sys_args.directed_interactions_filename
    args['data']['sources_filename'] = sources_filenmae_dict[sys_args.experiments_type]
    args['data']['terminals_filename'] = terminals_filenmae_dict[sys_args.experiments_type]
    args['data']['n_experiments'] = n_experiments

    # data read and filtering
    rng = np.random.RandomState(args['data']['random_seed'])

    network, directed_interactions, sources, terminals, id_to_degree = \
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

    deep_probs_list, deep_labels_list, d2d_2_probs_list, d2d_2_labels_list, d2d_probs_list, d2d_labels_list = [
        defaultdict(list) for x in range(6)]
    best_models = []
    deep_folds_stats, d2d_2_folds_stats, d2d_folds_stats = [], [], []

    for fold in range(n_folds):
        train_indexes, val_indexes, test_indexes = train_test_split(args['data']['split_type'],
                                                                    len(directed_interactions_pairs_list),
                                                                    args['train']['train_val_test_split'],
                                                                    random_state=rng)  # feature generation
        d2d_train_indexes = np.concatenate([train_indexes, val_indexes])

        # d2d evaluation
        sources_indexes = [[row_id_to_idx[gene_id] for gene_id in gene_set] for gene_set in sources.values()]
        terminals_indexes = [[row_id_to_idx[gene_id] for gene_id in gene_set] for gene_set in terminals.values()]
        pairs_indexes = [(col_id_to_idx[pair[0]], col_id_to_idx[pair[1]]) for pair in directed_interactions_pairs_list]
        features, deconstructed_features = generate_D2D_features_from_propagation_scores(propagation_scores,
                                                                                         pairs_indexes,
                                                                                         sources_indexes,
                                                                                         terminals_indexes)
        d2d_results_dict, model = eval_D2D(features[d2d_train_indexes], features[test_indexes],
                                           directed_interactions_source_type[test_indexes])
        d2d_2_results_dict, model = eval_D2D_2(deconstructed_features[d2d_train_indexes],
                                               deconstructed_features[test_indexes],
                                               directed_interactions_source_type[test_indexes])
        d2d_folds_stats.append({type: {x: xx for x, xx in values.items() if x in ['acc', 'auc']} for type, values in
                                d2d_results_dict.items()})
        d2d_2_folds_stats.append({type: {x: xx for x, xx in values.items() if x in ['acc', 'auc']} for type, values in
                                  d2d_2_results_dict.items()})
        train_dataset = LightDataset(row_id_to_idx, col_id_to_idx, propagation_scores,
                                     directed_interactions_pairs_list[train_indexes],
                                     sources, terminals, args['data']['normalization_method'],
                                     normalization_constants_dict, degree_feature_normalization_constants=None,
                                     pairs_source_type=directed_interactions_source_type, id_to_degree=id_to_degree)
        train_loader = DataLoader(train_dataset, batch_size=args['train']['train_batch_size'], shuffle=True,
                                  pin_memory=False, num_workers=sys_args.n_workers)

        degree_normalization_constants = {'lmbda': train_dataset.degree_normalizer.lmbda,
                                          'mean': train_dataset.degree_normalizer.lmbda,
                                          'std': train_dataset.degree_normalizer.std}
        val_dataset = LightDataset(row_id_to_idx, col_id_to_idx,
                                   propagation_scores, directed_interactions_pairs_list[val_indexes],
                                   sources, terminals, args['data']['normalization_method'],
                                   normalization_constants_dict, degree_normalization_constants,
                                   directed_interactions_source_type[val_indexes], id_to_degree)
        val_loader = DataLoader(val_dataset, batch_size=args['train']['test_batch_size'], shuffle=False,
                                pin_memory=False, num_workers=sys_args.n_workers)
        test_dataset = LightDataset(row_id_to_idx, col_id_to_idx, propagation_scores,
                                    directed_interactions_pairs_list[test_indexes],
                                    sources, terminals, args['data']['normalization_method'],
                                    normalization_constants_dict, degree_normalization_constants,
                                    directed_interactions_source_type[test_indexes], id_to_degree)
        test_loader = DataLoader(test_dataset, batch_size=args['train']['test_batch_size'], shuffle=False,
                                 pin_memory=False, )

        # train
        intermediate_loss_type = get_loss_function(args['train']['intermediate_loss_type'],
                                                   focal_gamma=args['train']['focal_gamma'])
        fold_models = []
        fold_aucs = []
        fold_stats = []
        for i in range(n_models_per_fold):
            # build models`
            deep_prop_model = DeepProp(args['model'], n_experiments)
            model = DeepPropClassifier(deep_prop_model)
            model.to(device=device)
            optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args['train']['learning_rate'])
            trainer = ClassifierTrainer(args['train']['n_epochs'], criteria=nn.CrossEntropyLoss(reduction='sum'),
                                        intermediate_criteria=intermediate_loss_type,
                                        intermediate_loss_weight=args['train']['intermediate_loss_weight'],
                                        optimizer=optimizer, eval_metric=None,
                                        eval_interval=args['train']['eval_interval'],
                                        device=device)
            train_stats, best_model = trainer.train(train_loader=train_loader, eval_loader=val_loader, model=model,
                                                    max_evals_no_improvement=args['train']['max_evals_no_imp'])
            fold_models.append(best_model)
            fold_aucs.append(train_stats['best_auc'])
            fold_stats.append(train_stats)
        deep_folds_stats.append(fold_stats[np.argmax(fold_aucs)])
        best_model = fold_models[np.argmax(fold_aucs)]
        best_models.append(best_model)
        deep_results = trainer.eval_by_source(best_model, test_loader, by_source_type=True, output_probs='True')
        for key in deep_results.keys():
            deep_probs_list[key].append(deep_results[key]['probs'][:, 1])
            deep_labels_list[key].append(deep_results[key]['labels'])

            d2d_probs_list[key].append(d2d_results_dict[key]['probs'][:, 1])
            d2d_labels_list[key].append(d2d_results_dict[key]['labels'])
            d2d_2_probs_list[key].append(d2d_2_results_dict[key]['probs'][:, 1])
            d2d_2_labels_list[key].append(d2d_2_results_dict[key]['labels'])

    final_stats = {'deep': {'final': {}, 'folds_stats': deep_folds_stats},
                   'd2d': {'final': {}, 'folds_stats': d2d_folds_stats},
                   'd2d_2': {'final': {}, 'folds_stats': d2d_2_folds_stats}}
    deep_probs_by_type, deep_labels_by_type, d2d_2_probs_by_type, d2d_2_labels_by_type, d2d_probs_by_type, d2d_labels_by_type = [
        defaultdict(list) for x in range(6)]
    for type in deep_results.keys():
        deep_probs_by_type[type] = np.hstack(deep_probs_list[type])
        deep_labels_by_type[type] = np.hstack(deep_labels_list[type])
        d2d_probs_by_type[type] = np.hstack(d2d_probs_list[type])
        d2d_labels_by_type[type] = np.hstack(d2d_labels_list[type])
        d2d_2_probs_by_type[type] = np.hstack(d2d_2_probs_list[type])
        d2d_2_labels_by_type[type] = np.hstack(d2d_2_labels_list[type])

        deep_precision, deep_recall, deep_thresholds = precision_recall_curve(deep_labels_by_type[type],
                                                                              deep_probs_by_type[type])
        deep_auc = auc(deep_recall, deep_precision)
        d2d_precision, d2d_recall, d2d_thresholds = precision_recall_curve(d2d_labels_by_type[type],
                                                                           d2d_probs_by_type[type])
        d2d_auc = auc(d2d_recall, d2d_precision)
        d2d_precision_2, d2d_recall_2, d2d_thresholds_2 = precision_recall_curve(d2d_2_labels_by_type[type],
                                                                                 d2d_2_probs_by_type[type])
        d2d_2_auc = auc(d2d_recall_2, d2d_precision_2)

        final_stats['deep']['final']['auc'] = deep_auc
        final_stats['d2d']['final']['auc'] = d2d_auc
        final_stats['d2d_2']['final']['auc'] = d2d_2_auc

        plt.plot([0, 1], [0.5, 0.5], '--', color=(0.8, 0.8, 0.8), label='random')
        plt.plot(deep_recall, deep_precision, 'o--', label='DeepSetPropagation (%0.2f)' % deep_auc, lw=2,
                 markersize=3)  # , markeredgecolor = 'dimgrey')
        plt.plot(d2d_recall, d2d_precision, 'o--', label='D2D (%0.2f)' % d2d_auc, lw=2,
                 markersize=3)  # , markeredgecolor = 'dimgrey')
        plt.plot(d2d_recall_2, d2d_precision_2, 'o--', label='D2D deconstructed (%0.2f)' % d2d_2_auc, lw=2,
                 markersize=3)  # , markeredgecolor = 'dimgrey')

        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(loc="lower right")
        plt.title('{}, {}, {} sources, {} folds, {}'.format(' '.join(args['data']['directed_interactions_filename']),
                                                            args['data']['sources_filename'].split('_')[-1],
                                                            n_experiments, n_folds, type))

        plt.savefig(path.join(output_file_path, 'auc_curve_{}'.format(type)))

        plt.close()
    for fold in range(n_folds):
        save_model(path.join(output_file_path, 'model_fold_{}'.format(fold)), best_models[fold])
    with open(path.join(output_file_path, 'args'), 'w') as f:
        json.dump(args, f, indent=4, separators=(',', ': '))
    with open(path.join(output_file_path, 'results'), 'w') as f:
        json.dump(final_stats, f, indent=4, separators=(',', ': '))


if __name__ == '__main__':
    n_folds = 3
    input_type = 'drug'
    load_prop = False
    save_prop = False
    n_exp = 2
    split = [0.66, 0.14, 0.2]
    interaction_type = ['KPI']
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
    parser.add_argument('-in', '--inter_file', dest='directed_interactions_filename', nargs='*', type=str,
                        help='KPI/STKE', default=interaction_type)
    parser.add_argument('-p', '--prop_file', dest='prop_scores_filename', type=str,
                        help='Name of prop score file(save/load)', default=prop_scores_filename)
    parser.add_argument('-f', '--n_folds', dest='n_folds', type=str,
                        help='Name of prop score file(save/load)', default=n_folds)
    parser.add_argument('-w', dest='n_workers', type=int,
                        help='number of dataloader workers', default=0)

    args = parser.parse_args()
    args.directed_interactions_filename = sorted(args.directed_interactions_filename)
    args.prop_scores_filename = args.experiments_type + '_' + '_'.join(args.directed_interactions_filename) + '_{}'.format(args.n_experiments)

    args.load_prop_scores =  True
    # args.save_prop_scores =  True
    run(args)
