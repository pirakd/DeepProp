import numpy as np
import networkx as nx
import math
import scipy
from os import path, makedirs
from datetime import datetime
import json
import pandas as pd


def balance_dataset(network, directed_interactions):
    graph = nx.from_pandas_edgelist(network.reset_index(), 0, 1, 'edge_score')
    directed_interactions = directed_interactions[directed_interactions.index.get_level_values(0).isin(graph) & directed_interactions.index.get_level_values(1).isin(graph)]
    degree = dict(graph.degree(weight='edge_score'))
    source_more_central = np.array([degree[s]>degree[t] for s, t in directed_interactions.index])
    larger_indexes = np.nonzero(source_more_central)[0]
    smaller_indexes = np.nonzero(1-source_more_central)[0]
    if larger_indexes.size > smaller_indexes.size:
        larger_indexes = np.random.choice(larger_indexes, smaller_indexes.size, replace=False)
    else:
        smaller_indexes = np.random.choice(smaller_indexes, larger_indexes.size, replace=False)
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


def read_priors(sources_filename, terminals_filename):
    source_priors = pd.read_table(sources_filename, header=None).groupby(0)[1].apply(set).to_dict()
    terminal_priors = pd.read_table(terminals_filename, header=None).groupby(0)[1].apply(set).to_dict()
    return source_priors, terminal_priors


def read_data(network_filename, directed_interaction_filename, sources_filename, terminals_filename):
    from gene_name_translator.gene_translator import GeneTranslator
    translator = GeneTranslator(verbosity=False)
    translator.load_dictionary()

    network = read_network(network_filename, translator)
    directed_interactions = read_directed_interactions(directed_interaction_filename, translator)
    merged_network =\
        pd.concat([network.drop(directed_interactions.index.intersection(network.index)), directed_interactions,])

    directed_interactions = balance_dataset(merged_network, directed_interactions)
    sources, terminals = read_priors(sources_filename, terminals_filename)
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


def generate_feature_columns(network, sources, terminals, indexes_to_keep, propagate_alpha, propagate_iterations, propagation_epsilon, n_exp):
    gene_indexes, matrix, num_genes = generate_propagate_data(network, propagate_alpha)
    gene1_indexes, gene2_indexes = map(lambda x: x, zip(*[[(gene_indexes[gene]) for gene in pair] for pair in network.index]))
    experiments = sorted(sources.keys() & terminals.keys())[:n_exp]

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


def get_pulling_func(pulling_func_name):
    from deep_learning.deep_utils import MaskedSum, MaskedMean

    if pulling_func_name == 'mean':
        return MaskedMean
    elif pulling_func_name == 'sum':
        return MaskedSum
    else:
        assert 0, '{} is not a vallid pulling operation function name'.format(pulling_func_name)


def get_root_path():
    return path.dirname(path.realpath(__file__))


def get_time():
    return datetime.today().strftime('%d_%m_%Y__%H_%M_%S')


def log_results(results_dict, results_path):
    time = get_time()
    file_path = path.join(results_path, time)
    with open(file_path, 'w') as f:
        json.dump(results_dict, f, indent=4, separators=(',', ': '))

