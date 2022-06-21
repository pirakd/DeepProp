
import pandas as pd
from os import path
import sys
sys.path.append(path.dirname(path.dirname(path.realpath(__file__))))
from os import makedirs
from deep_learning.data_loaders import LightDataset
from deep_learning.trainer import ClassifierTrainer
from torch import nn
from torch.utils.data import DataLoader
from utils import read_data, load_model, log_results, get_time, get_root_path, train_test_split,\
    gen_propagation_scores, get_loss_function, redirect_output
import torch
import numpy as np
import json
import argparse
import copy
import networkx as nx

def run(sys_args):
    root_path = get_root_path()
    output_folder = 'output'
    output_file_path = path.join(get_root_path(), output_folder, path.basename(__file__).split('.')[0], get_time())
    makedirs(output_file_path, exist_ok=True)
    redirect_output(path.join(output_file_path, 'log'))

    model_args_path = path.join(root_path, 'input', 'models', sys_args.model_name, 'args')
    model_results_path = path.join(root_path, 'input', 'models', sys_args.model_name, 'results')
    model_path = path.join(root_path, 'input', 'models', sys_args.model_name, 'model')

    with open(model_args_path, 'r') as f:
        args = json.load(f)
    with open(model_results_path, 'r') as f:
        normalization_constants_dicts = json.load(f)['normalization_constants_dicts']

    device = torch.device("cuda".format(sys_args.device) if torch.cuda.is_available() else "cpu")
    args['data']['load_prop_scores'] = sys_args.load_prop_scores
    args['data']['save_prop_scores'] = sys_args.save_prop_scores
    args['data']['prop_scores_filename'] = sys_args.prop_scores_filename
    args['train']['train_val_test_split'] = sys_args.train_val_test_split
    args['data']['directed_interactions_filename'] = sys_args.directed_interactions_filename
    model = load_model(model_path, args).to(device)
    rng = np.random.RandomState(args['data']['random_seed'])

    # data read
    network, directed_interactions, sources, terminals, id_to_degree =\
        read_data(args['data']['network_filename'], args['data']['directed_interactions_filename'],
                  args['data']['sources_filename'], args['data']['terminals_filename'],
                  args['data']['n_experiments'], args['data']['max_set_size'], rng, translate_genes=True)

    # random_indexes = np.random.choice(len(network.index), 100)
    # directed_interactions_pairs_list  = np.array(list(network.index))
    # n_experiments = len(sources.keys())
    # directed_interactions_source_type = np.array(['yeast_KPI' for x in range(len(directed_interactions_pairs_list))])

    # genes_ids_to_keep =/
    train_interactions_pairs_list = np.array(directed_interactions.index)
    train_set_genes = sorted(list(set([x for pair in train_interactions_pairs_list for x in pair])))
    graph = nx.from_pandas_edgelist(network.reset_index(), 0, 1, 'edge_score')
    genes_ids_to_keep = graph.nodes
    directed_interactions_pairs_list = np.array([(x[0], x[1]) for x in list(network.index) if (x[0] in genes_ids_to_keep and x[1] in genes_ids_to_keep)])
    genes_ids_to_keep = sorted(genes_ids_to_keep)
    directed_interactions_source_type = np.array(['KEGG' for x in range(len(directed_interactions_pairs_list))])

    # here we have to propagate all genes
    propagation_scores, row_id_to_idx, col_id_to_idx, _ = \
        gen_propagation_scores(args, network, sources, terminals, genes_ids_to_keep, directed_interactions_pairs_list, calc_normalization_constants=False)

    train_indexes, val_indexes, test_indexes = train_test_split(args['data']['split_type'],len(directed_interactions_pairs_list), args['train']['train_val_test_split'],
                                                                random_state=rng, directed_interactions=directed_interactions_pairs_list)

    test_dataset = LightDataset(row_id_to_idx, col_id_to_idx, propagation_scores,
                                directed_interactions_pairs_list[test_indexes], sources,
                                terminals, args['data']['normalization_method'],
                                samples_normalization_constants=normalization_constants_dicts['samples'],
                                degree_feature_normalization_constants= normalization_constants_dicts['degrees'],
                                pairs_source_type=directed_interactions_source_type[test_indexes],
                                id_to_degree=id_to_degree, train=False)

    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, pin_memory=False, num_workers=sys_args.n_workers)
    intermediate_loss_type = get_loss_function(args['train']['intermediate_loss_type'],
                                               focal_gamma=args['train']['focal_gamma'])
    trainer = ClassifierTrainer(args['train']['n_epochs'], criteria=nn.CrossEntropyLoss(), intermediate_criteria=intermediate_loss_type,
                                intermediate_loss_weight=args['train']['intermediate_loss_weight'],
                                optimizer=None, eval_metric=None, eval_interval=args['train']['eval_interval'], device='cpu')

    # check prediction on trained samples
    # trained_interaction = np.array(list(directed_interactions.index))
    # trained_interaction_source_type = np.array(['yeast_KPI' for x in range(len(trained_interaction))])
    # sanity_dataset =  LightDataset(row_id_to_idx, col_id_to_idx, propagation_scores,
    #                         trained_interaction, sources,
    #                         terminals, args['data']['normalization_method'],
    #                         samples_normalization_constants=normalization_constants_dicts['samples'],
    #                         degree_feature_normalization_constants= normalization_constants_dicts['degrees'],
    #                         pairs_source_type=trained_interaction_source_type,
    #                         id_to_degree=id_to_degree, train=False)
    # sanity_loader = DataLoader(sanity_dataset, batch_size=200, shuffle=False, pin_memory=False, )
    # results = trainer.predict(model, sanity_loader)
    #

    results = trainer.predict(model, test_loader)
    # turn predictions to a dict
    col_idx_to_id = {xx: x for x, xx in col_id_to_idx.items()}
    directed_edge_prob = {tuple([id for id in tuple(results['pairs'][i,:])]) : results['probs'][i,1] for i in range(results['probs'].shape[0])}
    # duplicate network

    network.pop('source')
    network_reverse = copy.deepcopy(network)
    reverse_indexes = np.array([[x[1], x[0]] for x in list(network.index)])
    network_reverse.index = pd.MultiIndex.from_arrays([reverse_indexes[:, 0], reverse_indexes[:, 1]])
    directed_network = pd.concat([network, network_reverse])
    directed_network['direction_prob'] = [directed_edge_prob.get((x[0],x[1]), 0) for x in list(directed_network.index)]
    directed_network.reset_index().to_csv(path.join(output_file_path, 'directed_network'), sep='\t', header=True, index=False)

    results_dict = {'n_experiments': model.n_experiments, 'model_path': model_path}
    with open(path.join(output_file_path, 'results'), 'w') as f:
        json.dump(results_dict, f, indent=4, separators=(',', ': '))
    with open(path.join(output_file_path, 'args'), 'w') as f:
        json.dump(args, f, indent=4, separators=(',', ': '))

if __name__ == '__main__':
    input_type = 'drug'
    n_exp = 5
    split = [0.99, 0, 0.01]

    interaction_type = sorted(['KPI', 'E3', 'EGFR', 'STKE', 'PDI'])
    interaction_type = sorted(['KEGG'])
    interaction_type = sorted(['KPI'])

    # interaction_type = 'yeast_KPI'
    device = 'cpu'
    model_name = 'drug_KEGG_5'
    # prop_scores_filename = 'yeast_KPI_direct'
    # prop_scores_filename = 'ovary_E3_EGFR_KPI_STKE_direct'
    prop_scores_filename = ''

    parser = argparse.ArgumentParser()
    parser.add_argument('-ex', '--ex_type', dest='experiments_type', type=str, help='name of experiment type(drug, colon, etc.)', default=input_type)
    parser.add_argument('-n,', '--n_exp', dest='n_experiments', type=int, help='num of experiments used (0 for all)', default=n_exp)
    parser.add_argument('-m,', '--model_name', type=str, help='name of saved model folder in input/models', default=model_name)
    parser.add_argument('-s', '--save_prop', dest='save_prop_scores',  action='store_true', default=False, help='Whether to save propagation scores')
    parser.add_argument('-l', '--load_prop', dest='load_prop_scores',  action='store_true', default=False, help='Whether to load prop scores')
    parser.add_argument('-sp', '--split', dest='train_val_test_split',  nargs=3, help='[train, val, test] sums to 1', default=split, type=float)
    parser.add_argument('-d', '--device', type=str, help='cpu or gpu number',  default=device)
    parser.add_argument('-in', '--inter_file', dest='directed_interactions_filename', type=str, help='KPI/STKE',
                        default=interaction_type)
    parser.add_argument('-p', '--prop_file', dest='prop_scores_filename', type=str,
                        help='Name of prop score file(save/load)', default=prop_scores_filename)
    parser.add_argument('-w', dest='n_workers', type=int,
                        help='number of dataloader workers', default=0)
    sys_args = parser.parse_args()
    sys_args.prop_scores_filename = sys_args.experiments_type + '_' + '_'.join(sys_args.directed_interactions_filename) + '_{}_direct'.format(sys_args.n_experiments)
    sys_args.model_name = sys_args.experiments_type + '_' + '_'.join(sys_args.directed_interactions_filename) + '_{}'.format(sys_args.n_experiments)

    # sys_args.save_prop_scores =  True
    # sys_args.load_prop_scores =  True

    run(sys_args)
