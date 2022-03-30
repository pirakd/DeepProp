from os import path
import sys
sys.path.append(path.dirname(path.dirname(path.realpath(__file__))))
from os import makedirs
from deep_learning.data_loaders import LightDataset
from deep_learning.models import DeepPropClassifier, DeepProp
from deep_learning.trainer import ClassifierTrainer
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from utils import read_data, log_results, get_time, get_root_path, train_test_split, get_loss_function,\
    gen_propagation_scores
import torch
import numpy as np
from presets import experiments_20, experiments_50, experiments_all
from scripts.scripts_utils import sources_filenmae_dict, terminals_filenmae_dict
import argparse


def run(sys_args):
    root_path = get_root_path()
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
    rng = np.random.RandomState(args['data']['random_seed'])

    # data read
    network, directed_interactions, sources, terminals, id_to_degree =\
        read_data(args['data']['network_filename'], args['data']['directed_interactions_filename'],
                  args['data']['sources_filename'], args['data']['terminals_filename'],
                  args['data']['n_experiments'], args['data']['max_set_size'], rng)

    n_experiments = len(sources.keys())
    directed_interactions_pairs_list = np.array(directed_interactions.index)
    directed_interactions_source_type = np.array(directed_interactions.source)
    genes_ids_to_keep = sorted(list(set([x for pair in directed_interactions_pairs_list for x in pair])))

    propagation_scores, row_id_to_idx, col_id_to_idx, normalization_constants_dict = \
        gen_propagation_scores(args, network, sources, terminals, genes_ids_to_keep, directed_interactions_pairs_list)

    # generating datasets
    train_indexes, val_indexes, test_indexes = train_test_split(len(directed_interactions_pairs_list), args['train']['train_val_test_split'],
                                                                random_state=rng)
    train_dataset = LightDataset(row_id_to_idx, col_id_to_idx, propagation_scores, directed_interactions_pairs_list[train_indexes], sources, terminals, args['data']['normalization_method'],
                                 normalization_constants_dict, directed_interactions_source_type[train_indexes], id_to_degree=id_to_degree)
    train_loader = DataLoader(train_dataset, batch_size=args['train']['train_batch_size'], shuffle=True, pin_memory=False)
    val_dataset = LightDataset(row_id_to_idx, col_id_to_idx, propagation_scores, directed_interactions_pairs_list[val_indexes], sources, terminals, args['data']['normalization_method'],
                               normalization_constants_dict, directed_interactions_source_type[val_indexes], id_to_degree)
    val_loader = DataLoader(val_dataset, batch_size=args['train']['test_batch_size'], shuffle=False, pin_memory=False)
    test_dataset = LightDataset(row_id_to_idx, col_id_to_idx, propagation_scores, directed_interactions_pairs_list[test_indexes], sources, terminals, args['data']['normalization_method'],
                                normalization_constants_dict, directed_interactions_source_type[test_indexes], id_to_degree)
    test_loader = DataLoader(test_dataset, batch_size=args['train']['test_batch_size'], shuffle=False, pin_memory=False,)

    # init model
    deep_prop_model = DeepProp(args['model'], n_experiments)
    model = DeepPropClassifier(deep_prop_model).to(device)

    # init train
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args['train']['learning_rate'])
    intermediate_loss_type = get_loss_function(args['train']['intermediate_loss_type'],
                                               focal_gamma=args['train']['focal_gamma'])
    trainer = ClassifierTrainer(args['train']['n_epochs'], criteria=nn.CrossEntropyLoss(reduction='sum'), intermediate_criteria=intermediate_loss_type,
                                intermediate_loss_weight=args['train']['intermediate_loss_weight'],
                                optimizer=optimizer, eval_metric=None, eval_interval=args['train']['eval_interval'], device=device)

    # train
    train_stats, best_model = trainer.train(train_loader=train_loader, eval_loader=val_loader, model=model,
                                            max_evals_no_improvement=args['train']['max_evals_no_imp'])

    if len(test_dataset):
        test_results_dict = \
            trainer.eval_by_source(best_model, test_loader)
        print(test_results_dict)
    else:
        test_results_dict = {}

    results_dict = {'train_stats': train_stats, 'test_stats': test_results_dict, 'n_experiments': n_experiments}
    log_results(output_file_path,  args, results_dict, best_model)


if __name__ == '__main__':
    input_type = 'drug'
    load_prop = False
    save_prop = False
    n_exp = 2
    split = [0.66, 0.14, 0.2]
    interaction_type = ['KPI', 'STKE']
    device = 'cpu'
    prop_scores_filename = 'drug_KPI_STKE'

    parser = argparse.ArgumentParser()
    parser.add_argument('-ex', '--ex_type', dest='experiments_type', type=str, help='name of experiment type(drug, colon, etc.)', default=input_type)
    parser.add_argument('-n,', '--n_exp', dest='n_experiments', type=int, help='num of experiments used (0 for all)', default=n_exp)
    parser.add_argument('-s', '--save_prop', dest='save_prop_scores',  action='store_true', default=False, help='Whether to save propagation scores')
    parser.add_argument('-l', '--load_prop', dest='load_prop_scores',  action='store_true', default=False, help='Whether to load prop scores')
    parser.add_argument('-sp', '--split', dest='train_val_test_split',  nargs=3, help='[train, val, test] sums to 1', default=split, type=float)
    parser.add_argument('-d', '--device', type=str, help='cpu or gpu number',  default=device)
    parser.add_argument('-in', '--inter_file', dest='directed_interactions_filename', type=str, help='KPI/STKE',
                        default=interaction_type)
    parser.add_argument('-p', '--prop_file', dest='prop_scores_filename', type=str,
                        help='Name of prop score file(save/load)', default=prop_scores_filename)

    args = parser.parse_args()
    args.load_prop_scores = True
    # args.save_prop_scores =  True
    run(args)
