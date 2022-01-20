from os import path
from utils import read_data,  get_root_path
import pandas as pd
from gene_name_translator.gene_translator import GeneTranslator
import networkx as nx
import numpy as np

root_path = get_root_path()
input_file = path.join(root_path, 'input')
NETWORK_FILENAME = path.join(input_file, 'networks', "H_sapiens.net")
DIRECTED_INTERACTIONS_FILENAME = path.join(input_file, 'directed_interactions', "KPI_dataset")
SOURCES_FILENAME = path.join(input_file, 'priors', "drug_targets.txt")
TERMINALS_FILENAME = path.join(input_file, 'priors', "drug_expressions.txt")


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

def read_network(network_filename, translator= None):

    network = pd.read_table(network_filename, header=None, usecols=[0, 1, 2], index_col=[0, 1]).rename(
        columns={2: 'edge_score'})
    if translator:
        gene_ids = set(network.index.get_level_values(0)).union(set(network.index.get_level_values(1)))
        up_do_date_ids = translator.translate(gene_ids, 'entrez_id', 'entrez_id')
        network.rename(index=up_do_date_ids, inplace=True)
    return network


def read_directed_interactions(directed_interactions_filename):
    directed_interactions = pd.read_table(directed_interactions_filename, usecols=['GENE', 'SUB_GENE'])
    genes = pd.unique(directed_interactions[['GENE', 'SUB_GENE']].values.ravel())
    symbol_to_entrez = translator.translate(genes, 'symbol', 'entrez_id')
    has_translation = directed_interactions['GENE'].isin(symbol_to_entrez) & directed_interactions['SUB_GENE'].isin(symbol_to_entrez)
    not_self_edge = directed_interactions['GENE'].ne(directed_interactions['SUB_GENE'])
    directed_interactions = directed_interactions[has_translation & not_self_edge]
    directed_interactions.replace(symbol_to_entrez, inplace=True)
    directed_interactions['edge_score'] = 0.8
    directed_interactions.rename(columns={'GENE':'KIN_GENE'}, inplace=True)
    directed_interactions.index = pd.MultiIndex.from_arrays(directed_interactions[['KIN_GENE', 'SUB_GENE']].values.T)
    return directed_interactions[['edge_score']]


def read_priors(sources_filename, terminals_filename):
    source_priors = pd.read_table(sources_filename, header=None).groupby(0)[1].apply(set).to_dict()
    terminal_priors = pd.read_table(terminals_filename, header=None).groupby(0)[1].apply(set).to_dict()
    return source_priors, terminal_priors


translator = GeneTranslator(verbosity=False)
translator.load_dictionary()
network = read_network(NETWORK_FILENAME, translator)
directed_interactions = read_directed_interactions(DIRECTED_INTERACTIONS_FILENAME)
merged_network = pd.concat([directed_interactions, network.drop(directed_interactions.index.intersection(network.index))])
directed_interactions = balance_dataset(network, directed_interactions)