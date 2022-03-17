import numpy as np
import networkx as nx
import math
import scipy
from os import path
from datetime import datetime
import pickle
import json
import pandas as pd
import torch
from deep_learning.deep_utils import FocalLoss
from tqdm import tqdm


def balance_dataset(network, directed_interactions, rng):
    graph = nx.from_pandas_edgelist(network.reset_index(), 0, 1, 'edge_score')
    directed_interactions = directed_interactions[directed_interactions.index.get_level_values(0).isin(graph) & directed_interactions.index.get_level_values(1).isin(graph)]
    degree = dict(graph.degree(weight='edge_score'))
    source_more_central = np.array([degree[s] > degree[t] for s, t in directed_interactions.index])
    larger_indexes = np.nonzero(source_more_central)[0]
    smaller_indexes = np.nonzero(1-source_more_central)[0]
    if larger_indexes.size > smaller_indexes.size:
        larger_indexes = rng.choice(larger_indexes, smaller_indexes.size, replace=False)
    else:
        smaller_indexes = rng.choice(smaller_indexes, larger_indexes.size, replace=False)
    return directed_interactions.iloc[np.sort(np.hstack([smaller_indexes, larger_indexes]))]


def read_network(network_filename, translator):
    network = pd.read_table(network_filename, header=None, usecols=[0, 1, 2], index_col=[0, 1]).rename(
        columns={2: 'edge_score'})
    if translator:
        gene_ids = set(network.index.get_level_values(0)).union(set(network.index.get_level_values(1)))
        up_do_date_ids = translator.translate(gene_ids, 'entrez_id', 'entrez_id')
        network.rename(index=up_do_date_ids, inplace=True)
    return network

def read_directed_interactions(directed_interactions_filename, gene_translator):
    directed_interactions = pd.read_table(directed_interactions_filename, usecols=['GENE', 'SUB_GENE'])
    genes = pd.unique(directed_interactions[['GENE', 'SUB_GENE']].values.ravel())
    symbol_to_entrez = gene_translator.translate(genes, 'symbol', 'entrez_id')
    has_translation = directed_interactions['GENE'].isin(symbol_to_entrez) & directed_interactions['SUB_GENE'].isin(symbol_to_entrez)
    not_self_edge = directed_interactions['GENE'].ne(directed_interactions['SUB_GENE'])
    directed_interactions = directed_interactions[has_translation & not_self_edge]
    directed_interactions.replace(symbol_to_entrez, inplace=True)
    directed_interactions['edge_score'] = 0.8
    directed_interactions.rename(columns={'GENE':'KIN_GENE'}, inplace=True)
    directed_interactions.index = pd.MultiIndex.from_arrays(directed_interactions[['KIN_GENE', 'SUB_GENE']].values.T)
    directed_interactions = directed_interactions[~directed_interactions.index.duplicated(keep='first')]

    return directed_interactions[['edge_score']]


def read_priors(sources_filename, terminals_filename, translator=None):
    source_priors = pd.read_table(sources_filename, header=None).groupby(0)[1].apply(set)
    terminal_priors = pd.read_table(terminals_filename, header=None).groupby(0)[1].apply(set)

    if translator:
        def translate(gene_set):
            translated_genes = translator.translate(gene_set, 'entrez_id', 'entrez_id')
            filtered_translated_genes = set([translated_genes[gene] for gene in gene_set if gene in translated_genes])
            return filtered_translated_genes
        source_priors = source_priors.apply(translate)
        terminal_priors = terminal_priors.apply(translate)
    source_priors = source_priors.to_dict()
    terminal_priors = terminal_priors.to_dict()
    return source_priors, terminal_priors


def read_data(network_filename, directed_interaction_filename, sources_filename, terminals_filename, n_exp, max_set_size, rng):
    # set paths
    root_path = get_root_path()
    input_file = path.join(root_path, 'input')
    network_file_path = path.join(input_file, 'networks', network_filename)
    directed_interaction_file_path = path.join(input_file, 'directed_interactions', directed_interaction_filename)
    sources_file_path = path.join(input_file, 'priors', sources_filename)
    terminals_file_path = path.join(input_file, 'priors', terminals_filename)

    # load gene translator
    from gene_name_translator.gene_translator import GeneTranslator
    translator = GeneTranslator(verbosity=False)
    translator.load_dictionary()

    # do all the reading stuff
    network = read_network(network_file_path, translator)
    directed_interactions = read_directed_interactions(directed_interaction_file_path, translator)
    merged_network =\
        pd.concat([network.drop(directed_interactions.index.intersection(network.index)), directed_interactions,])
    directed_interactions = balance_dataset(merged_network, directed_interactions, rng)
    sources, terminals = read_priors(sources_file_path, terminals_file_path, translator)

    # constrain to network's genes
    unique_genes = set((np.array(merged_network.index.to_list()).ravel()))
    sources = {exp_name: set([gene for gene in genes if gene in unique_genes]) for exp_name, genes in sources.items()}
    terminals = {exp_name: set([gene for gene in genes if gene in unique_genes]) for exp_name, genes in terminals.items()}

    # filter large sets
    sources = {exp_name: values for exp_name, values in sources.items() if len(values) <= max_set_size}
    terminals = {exp_name: values for exp_name, values in terminals.items() if len(values) <= max_set_size}

    # filter experiments that do not have a source and a terminal set
    filtered_experiments = sorted(sources.keys() & terminals.keys())

    # choose only a subset of the experiments
    if isinstance(n_exp, int):
        filtered_experiments = filtered_experiments[:n_exp]
    elif n_exp == 'all':
        pass
    else:
        assert 0, 'wrong input in args[data][n_experiments]'
    sources = {exp_name: sources[exp_name] for exp_name in filtered_experiments}
    terminals = {exp_name: terminals[exp_name] for exp_name in filtered_experiments}

    return merged_network, directed_interactions, sources, terminals


def generate_similarity_matrix(graph, propagate_alpha):
    genes = sorted(graph.nodes)
    matrix = nx.to_scipy_sparse_matrix(graph, genes, weight=2)
    norm_matrix = scipy.sparse.diags(1 / np.sqrt(matrix.sum(0).A1))
    matrix = norm_matrix * matrix * norm_matrix
    return propagate_alpha * matrix, genes


def propagate(seeds, matrix, gene_indexes, num_genes, propagate_alpha, propagate_iterations, propagate_epsilon):
    F_t = np.zeros(num_genes)
    F_t[[gene_indexes[seed] for seed in seeds if seed in gene_indexes]] = 1
    Y = (1 - propagate_alpha) * F_t

    for _ in range(propagate_iterations):
        F_t_1 = F_t
        F_t = matrix.dot(F_t_1) + Y

        if math.sqrt(scipy.linalg.norm(F_t_1 - F_t)) < propagate_epsilon:
            break
    return F_t


def generate_propagate_data(network, propagate_alpha):
    graph = nx.from_pandas_edgelist(network.reset_index(), 0, 1, 'edge_score')
    matrix, genes = generate_similarity_matrix(graph, propagate_alpha)
    num_genes = len(genes)
    gene_indexes = dict([(gene, index) for (index, gene) in enumerate(genes)])
    return gene_indexes, matrix, num_genes


def generate_feature_columns(network, sources, terminals, indexes_to_keep, propagate_alpha, propagate_iterations, propagation_epsilon):
    gene_indexes, matrix, num_genes = generate_propagate_data(network, propagate_alpha)
    gene1_indexes, gene2_indexes = map(lambda x: x, zip(*[[(gene_indexes[gene]) for gene in pair] for pair in network.index]))
    experiments = sources.keys()

    def generate_column(experiment):
        source_scores = np.array([propagate([s], matrix, gene_indexes, num_genes, propagate_alpha, propagate_iterations,
                                            propagation_epsilon) for s in sources[experiment]]).T
        terminal_scores = np.array([propagate([t], matrix, gene_indexes, num_genes,  propagate_alpha, propagate_iterations,
                                            propagation_epsilon) for t in terminals[experiment]]).T
        source_gene_1, source_gene_2 = [source_scores[gene1_indexes, :], source_scores[gene2_indexes, :]]
        source_features = np.concatenate([source_gene_1[..., np.newaxis],  source_gene_2[..., np.newaxis,]], axis=2)[indexes_to_keep, ...]
        terminal_gene_1, terminal_gene_2 = [terminal_scores[gene1_indexes, :], terminal_scores[gene2_indexes, :]]
        terminal_features = np.concatenate([terminal_gene_1[..., np.newaxis],  terminal_gene_2[..., np.newaxis]], axis=2)[indexes_to_keep, ...]

        return source_features, terminal_features

    source_features, terminal_features = [], []
    for experiment in experiments:
        curr_source_features, curr_terminal_features = generate_column(experiment)
        source_features.append(curr_source_features)
        terminal_features.append(curr_terminal_features)

    return source_features, terminal_features


def generate_raw_propagation_scores(network, sources, terminals, genes_ids_to_keep, propagate_alpha, propagate_iterations, propagation_epsilon):
    all_source_terminal_genes = sorted(list(set.union(set.union(*sources.values()), set.union(*terminals.values()))))
    gene_id_to_idx, matrix, num_genes = generate_propagate_data(network, propagate_alpha)
    gene_idx_to_id = {xx:x for x,xx in gene_id_to_idx.items()}
    genes_idxs_to_keep = [gene_id_to_idx[id] for id in genes_ids_to_keep]

    propagation_scores = np.array([propagate([s], matrix, gene_id_to_idx, num_genes, propagate_alpha, propagate_iterations,
                                        propagation_epsilon) for s in tqdm(all_source_terminal_genes,
                                                                           desc='propagating scores',
                                                                           total=len(all_source_terminal_genes))])

    propagation_scores = propagation_scores[:, genes_idxs_to_keep]
    col_id_to_idx = {gene_idx_to_id[idx]: i for i,idx in enumerate(genes_idxs_to_keep)}
    row_id_to_idx = {id:i for i, id in enumerate(all_source_terminal_genes)}

    return propagation_scores, row_id_to_idx, col_id_to_idx


def normalize_features(source_features, terminal_features, eps=1e-8):
    source_array =[]
    terminal_array = []
    for arr_idx in range(len(source_features)):
        source_array.append(source_features[arr_idx].ravel())
        terminal_array.append(terminal_features[arr_idx].ravel())
    source_array, terminal_array = np.hstack(source_array), np.hstack(terminal_array)
    source_mean, terminal_mean = np.mean(source_array), np.mean(terminal_array)
    source_std, terminal_std = np.std(source_array), np.std(terminal_array)
    for arr_idx in range(len(source_features)):
        source_features[arr_idx] = ((source_features[arr_idx] - source_mean) / source_std) +eps
        terminal_features[arr_idx] = ((terminal_features[arr_idx] - terminal_mean) / terminal_std) +eps

    return source_features, terminal_features


def get_normalization_constants(pairs_indexes, source_indexes, terminal_indexes, propagation_scores):
    """
    Acording to Welford's algorithm for calculating variance, See attached PDF for derivation.
    """
    total_source_elements, new_source_elements = 0, 0
    total_terminal_elements, new_terminal_elements = 0, 0
    terminal_mean, terminal_S = 0, 0
    source_mean, source_S = 0, 0

    for pair in pairs_indexes:
        for exp_idx in range(len(source_indexes)):
            source_feature = propagation_scores[:, pair][source_indexes[exp_idx], :].ravel()
            terminal_feature = propagation_scores[:, pair][terminal_indexes[exp_idx], :].ravel()

            new_source_elements, new_terminal_elements = len(source_feature), len(terminal_feature)
            total_source_elements += new_source_elements
            total_terminal_elements += new_terminal_elements

            source_delta_mean = source_feature-source_mean
            source_mean += np.sum(source_delta_mean) / total_source_elements
            source_S += np.sum((source_feature - source_mean) * (source_delta_mean))

            terminal_delta_mean = terminal_feature-terminal_mean
            terminal_mean += np.sum(terminal_delta_mean) / total_terminal_elements
            terminal_S += np.sum((terminal_feature - terminal_mean) * terminal_delta_mean)


    source_std = np.sqrt(source_S/(total_source_elements-1))
    terminal_std = np.sqrt(terminal_S/(total_terminal_elements-1))
    return {'source_mean':source_mean, 'source_std':source_std,
            'terminal_mean':terminal_mean, 'terminal_std':terminal_std}


def train_test_split(n_samples, train_test_ratio, random_state:np.random.RandomState=None):
    if random_state:
        split = random_state.choice([0, 1, 2], n_samples, replace=True, p=train_test_ratio )
    else:
        split = np.random.choice([0, 1, 2], n_samples, replace=True, p=train_test_ratio, )
    train_indexes = np.nonzero(split == 0)[0]
    val_indexes = np.nonzero(split == 1)[0]
    test_indexes = np.nonzero(split == 2)[0]

    return train_indexes, val_indexes, test_indexes


def get_pulling_func(pulling_func_name):
    from deep_learning.deep_utils import MaskedSum, MaskedMean

    if pulling_func_name == 'mean':
        return MaskedMean
    elif pulling_func_name == 'sum':
        return MaskedSum
    else:
        assert 0, '{} is not a vallid pulling operation function name'.format(pulling_func_name)


def get_loss_function(loss_name, **kwargs):
    if loss_name == 'BCE':
        return torch.nn.BCEWithLogitsLoss()
    elif loss_name == 'FOCAL':
        return FocalLoss(gamma=kwargs['focal_gamma'])
    else:
        assert 0, '{} is not a valid loss name'.format(loss_name)


def get_root_path():
    return path.dirname(path.realpath(__file__))


def get_time():
    return datetime.today().strftime('%d_%m_%Y__%H_%M_%S')


def log_results(results_dict, results_path):
    time = get_time()
    file_path = path.join(results_path, time)
    with open(file_path, 'w') as f:
        json.dump(results_dict, f, indent=4, separators=(',', ': '))


def save_propagation_score(propagation_scores, normalization_constants,
                           row_id_to_idx, col_id_to_idx, propagation_args, data_args, filename):
    save_dir = path.join(get_root_path(), 'input', 'propagation_scores', filename)
    save_dict = {'propagation_args': propagation_args, 'row_id_to_idx': row_id_to_idx, 'col_id_to_idx': col_id_to_idx,
                 'propagation_scores': propagation_scores, 'normalization_constants': normalization_constants,
                 'data_args': data_args, 'data': get_time()}
    save_pickle(save_dict, save_dir)
    return save_dir


def save_pickle(obj, save_dir):
    with open(save_dir, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(load_dir):
    with open(load_dir, 'rb') as f:
        obj = pickle.load(f)
    return obj
