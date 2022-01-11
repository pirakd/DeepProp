import pandas
import numpy as np
import networkx
import math
import scipy
from functools import partial


def read_network(network_filename):
    return pandas.read_table(network_filename, header=None, usecols=[0,1,2], index_col=[0,1])


def read_directed_interactions(directed_interactions_filename, type=None):
    directed_interactions = pandas.read_table(directed_interactions_filename, header=None, skiprows=1, usecols=[0,1,2,3], index_col=[0,1])
    if type:
        directed_interactions = directed_interactions[directed_interactions[2] == 'TRUE_' + type]
    else:
        directed_interactions = directed_interactions[directed_interactions[2] != "biogrid"]
    return directed_interactions[[3]].rename(columns={3:2})


def read_priors(priors_filename):
    return pandas.read_table(priors_filename, header=None).groupby(0)[1].apply(set).to_dict()


def read_data(network_filename, directed_interaction_filename, sources_filename, terminals_filename, type=None):
    network = read_network(network_filename)
    directed_interactions = read_directed_interactions(directed_interaction_filename, type)
    sources = read_priors(sources_filename)
    terminals = read_priors(terminals_filename)
    return network, directed_interactions, sources, terminals


def generate_similarity_matrix(graph, propagate_alpha):
    genes = sorted(graph.nodes)
    matrix = networkx.to_scipy_sparse_matrix(graph, genes, weight=2)
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
    graph = networkx.from_pandas_edgelist(network.reset_index(), 0, 1, 2)
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
