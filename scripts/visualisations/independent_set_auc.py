import pandas as pd
import numpy as np
from gene_name_translator.gene_translator import GeneTranslator
from collections import defaultdict
from utils import read_directed_interactions, read_network, balance_dataset, get_root_path, redirect_output, get_time
import matplotlib.pyplot as plt
import copy
import seaborn as sns
import networkx as nx
from sklearn.metrics import precision_recall_curve, auc, roc_curve
from scripts.scripts_utils import model_colors
from os import path, makedirs
network_filename = 'H_sapiens.net'
fontsize= 40
fontsize_2 = 24
fontsize_3 = 22

directions_predictions_files = {'breast': 'breast',
                                'ovary':'ovary',
                                'AML':'AML',
                                'colon':'colon',
                                'drug': 'drug',
                                'd2d_breast': 'd2d_breast',
                                'd2d_AML': 'd2d_AML',
                                'd2d_colon': 'd2d_colon',
                                'd2d_ovary': 'd2d_ovary',
                                'd2d_drug': 'd2d_drug'}
directions_predictions_files = {'breast': '23_06_2022__20_31_55',
                                'ovary':'23_06_2022__19_50_57',
                                'AML':'23_06_2022__19_50_30',
                                'colon':'25_06_2022__12_28_46',
                                'drug': '23_06_2022__19_50_08',
                                'd2d_breast': '25_06_2022__12_30_55',
                                'd2d_AML': '23_06_2022__18_56_52',
                                'd2d_colon': '25_06_2022__12_31_12',
                                'd2d_ovary': '23_06_2022__18_56_40',
                                'd2d_drug':'23_06_2022__18_07_16'}
prediction_types = ['ovary', 'AML', 'colon', 'breast']
root_path = get_root_path()

predction_folder = path.join(root_path, 'input', 'predicted_interactions')

# set paths
output_folder = 'output'
output_file_path = path.join(get_root_path(), output_folder, path.basename(__file__).split('.')[0], get_time())
makedirs(output_file_path, exist_ok=True)
redirect_output(path.join(output_file_path, 'log'))
input_file = path.join(root_path, 'input')
network_file_path = path.join(input_file, 'networks', network_filename)
directed_interaction_folder = path.join(input_file, 'directed_interactions')
path_linker_net_path = path.join(input_file,'networks', '2015pathlinker-weighted.txt')
deep_edge_probs_path = path.join(input_file, 'oriented_networks', '26_06_2022__14_17_41', 'edge_probs_ratio')
d2d_edge_probs_path = path.join(input_file, 'oriented_networks' , 'edge_probs_ratio_d2d')
uniprot_to_entrez_file_path = path.join(input_file, 'other', 'HUMAN_9606_idmapping.txt')

directed_interaction_filename = ['STKE', 'KPI', 'PDI', 'E3', 'EGFR']
translator = GeneTranslator()
translator.load_dictionary()
rng= np.random.RandomState(0)
unfiltered_directed_interactions = read_directed_interactions(directed_interaction_folder, directed_interaction_filename, translator)
network = read_network(network_file_path, translator)
sorted_interaction_df = copy.deepcopy(unfiltered_directed_interactions)
sorted_interaction_index = np.sort(np.array([list(x) for x in unfiltered_directed_interactions.index]), axis=1)
sorted_interaction_df.index = pd.MultiIndex.from_tuples(
    [(sorted_interaction_index[x, 0], sorted_interaction_index[x, 1]) for x in
     range(sorted_interaction_index.shape[0])], names=[0, 1])
sorted_interaction_df = sorted_interaction_df[~sorted_interaction_df.index.duplicated()]
merged_network = \
    pd.concat([network.drop(sorted_interaction_df.index.intersection(network.index)), sorted_interaction_df])
merged_graph = nx.from_pandas_edgelist(merged_network.reset_index(), 0, 1, 'edge_score')
directed_interactions = balance_dataset(merged_graph, unfiltered_directed_interactions, rng)
directed_interactions = set(list(directed_interactions.index))

df = pd.read_csv(path_linker_net_path, delimiter='\t', comment='#', header=None)
# kegg_df = df[df[3].str.contains('KEGG')]
# kegg_edges = kegg_df[[0, 1]].values.tolist()
# df = df[df[2] >= 0.75]
edges = df[[0, 1]].values.tolist()
forward_edges = set([tuple(x) for x in edges])
backward_edges = set([(x[1],x[0]) for x in edges])
directed_edge = list(forward_edges.difference(backward_edges))
with open(uniprot_to_entrez_file_path, 'r') as f:
    lines = f.readlines()
protein_dict = defaultdict(lambda: {'GeneID':'', 'Gene_Name':''})
for line in lines:
    protein, source, value = line.strip().split(sep='\t')
    if source == 'Gene_Name' or source == 'GeneID':
        protein_dict[protein][source] = value

all_genes = set(np.array(list(directed_edge)).ravel())
gene_ids = {}
count =0
for x in all_genes:
    if protein_dict[x]['GeneID'] != '':
        gene_ids[x] = int(protein_dict[x]['GeneID'])
    elif protein_dict[x]['Gene_Name'] != '':
        translated_gene = translator.translate([protein_dict[x]['Gene_Name']], 'symbol', 'entrez_id')[protein_dict[x]['Gene_Name']]
        if translated_gene:
            gene_ids[x] = translated_gene
    else:
        count+=1
# df.replace(gene_ids, inplace=True)
# df = df.set_index([0,1])
translated_edges = []
for edge in directed_edge:
    if edge[0] in gene_ids and edge[1] in gene_ids:
        translated_edges.append((gene_ids[edge[0]], gene_ids[edge[1]]))

translated_edges = set(translated_edges)
# df = df.loc[translated_edges]
flipped_translated_edge = set([(x[1], x[0]) for x in translated_edges])
all_dataset_edges = flipped_translated_edge.union(translated_edges)


predicted_edges = []
predictions = []
for n, name in enumerate(prediction_types):
    prediction_file_path = path.join(predction_folder, directions_predictions_files[name], 'directed_network')
    prediction = pd.read_csv(prediction_file_path, sep='\t', index_col=[0,1])
    predictions_dict = prediction[['direction_prob']].to_dict()['direction_prob']
    predicted_edges = {x: predictions_dict[x]/(predictions_dict[(x[1],x[0])] + 1e-12) for x in predictions_dict.keys()}
    prediction['direction_prob'].loc[predicted_edges.keys()] = list(predicted_edges.values())
    predictions.append(prediction)

merged_predictions = pd.concat(predictions)
merged_predictions['direction_prob'] = np.log(merged_predictions['direction_prob']+ 1e-9)
merged_predictions['direction_prob'] = merged_predictions.groupby(level=[0,1])['direction_prob'].mean()
merged_predictions = merged_predictions.reset_index().drop_duplicates(keep='first').set_index(keys=['0', '1'])

# edge_probs = pd.read_csv(deep_edge_probs_path, sep='\t', index_col=[0, 1])
edges = merged_predictions.index.to_numpy()
scores = merged_predictions['direction_prob'].to_numpy()
unfiltered_directed_interactions = set(unfiltered_directed_interactions.index.tolist())
indexes_to_keep = [x for x, xx in enumerate(edges) if (not (xx in flipped_translated_edge and xx in unfiltered_directed_interactions) and xx in all_dataset_edges and scores[x]>-15)]
edges = edges[indexes_to_keep]
scores = scores[indexes_to_keep]
labels = np.array([1 if x in translated_edges else 0 for x in edges])

d2d_edge_probs = pd.read_csv(d2d_edge_probs_path, sep='\t', index_col=[0,1])
d2d_edges = d2d_edge_probs.index.to_numpy()
d2d_scores = d2d_edge_probs['direction_prob'].to_numpy()
indexes_to_keep = [x for x, xx in enumerate(d2d_edges) if (not (xx in flipped_translated_edge and xx in unfiltered_directed_interactions) and xx in all_dataset_edges  and d2d_scores[x]>-15)]
d2d_edges = d2d_edges[indexes_to_keep]
d2d_scores = d2d_scores[indexes_to_keep]
d2d_labels = np.array([1 if x in translated_edges else 0 for x in d2d_edges])

sorted_indexes = np.argsort(scores)[::-1]
sorted_labels = labels[sorted_indexes]
ranks = np.argsort(np.argsort(scores))[sorted_indexes]
positive_ranks = ranks[np.nonzero(sorted_labels)[0]]
negative_ranks = ranks[np.nonzero(1-sorted_labels)[0]]

d2d_sorted_indexes = np.argsort(d2d_scores)[::-1]
d2d_sorted_labels = d2d_labels[d2d_sorted_indexes]
d2d_ranks = np.argsort(np.argsort(d2d_scores))[d2d_sorted_indexes]
d2d_positive_ranks = d2d_ranks[np.nonzero(d2d_sorted_labels)[0]]
d2d_negative_ranks = d2d_ranks[np.nonzero(1-d2d_sorted_labels)[0]]

d2d_precision, d2d_recall, _ = precision_recall_curve(d2d_sorted_labels, d2d_ranks)
d2d_directed_auc = auc(d2d_recall, d2d_precision)

fpr, tpr, thresholds = roc_curve(d2d_sorted_labels, d2d_ranks)
roc_auc = auc(fpr, tpr)

precision, recall, _ = precision_recall_curve(labels, scores)
directed_auc = auc(recall, precision)

fpr, tpr, thresholds = roc_curve(sorted_labels, ranks)
roc_auc = auc(fpr, tpr)

#predict separately for each model
# plt.plot([0, 1], [0.5, 0.5], '--', color=(0.8, 0.8, 0.8), label='random')
# plt.plot(recall, precision,
#          'o--', label='D-D2D (%0.2f)' % directed_auc, lw=2,
#          markersize=3)  # , markeredgecolor = 'dimgrey')
# plt.plot(d2d_recall, d2d_precision, 'o--',
#          label='D2D (%0.2f)' % d2d_directed_auc , lw=2,
#          markersize=3, model_color=model_colors)  # , markeredgecolor = 'dimgrey')
fig, ax = plt.subplots()
sns.lineplot(x=[0, 1], y=[0.5, 0.5], dashes=True, color=model_colors['random'], label='Random (0.50)', lw=6, ci=None, ax=ax)
sns.lineplot(x=recall,
             y=precision,
             markers='o--', color=model_colors['deep'],
             label="D'OR (%0.2f)" % directed_auc, lw=6,
             markersize=3, ci=None, ax=ax)
sns.lineplot(x=d2d_recall,
             y=d2d_precision, markers='o--', color=model_colors['d2d'],
             label='D2D (%0.2f)' % d2d_directed_auc, lw=6, markersize=3, ci=None, ax=ax)
ax.tick_params(axis='both', which='major', labelsize=fontsize_3)
plt.grid(True)
plt.xlim([0, 1])
plt.ylim([0.5, 1])
plt.xlabel('Recall', fontsize=fontsize_2)
plt.ylabel('Precision', fontsize=fontsize_2)
plt.legend(loc="upper right", fontsize=fontsize_2)
# plt.title('Performance over PathLinker Dataset', fontsize= fontsize)
fig.set_size_inches(12, 8)
plt.tight_layout()
plt.savefig(path.join(output_file_path,'aucpr_curve'))

fig, ax = plt.subplots()
sns.lineplot(x=recall,
             y=precision,
             markers='o--',
             label="D'OR consensus(%0.2f)" % directed_auc, lw=4,
             markersize=3, ci=None, ax=ax, alpha=0.8)

aucs= []
for p in range(len(predictions)):
    edges = predictions[p].index.to_numpy()
    score = predictions[p]['direction_prob'].to_numpy()
    indexes_to_keep = [x for x, xx in enumerate(edges) if (not (
                xx in flipped_translated_edge and xx in unfiltered_directed_interactions) and xx in all_dataset_edges and
                                                           score[x] > -15)]
    edges = edges[indexes_to_keep]
    score = score[indexes_to_keep]
    labels = np.array([1 if x in translated_edges else 0 for x in edges])

    single_precision, single_recall, _ = precision_recall_curve(labels, score)
    aucs.append(auc(single_recall, single_precision))
    sns.lineplot(x=single_recall,
                 y=single_precision,
                 markers='o--',
                 label="D'OR {}({:.2f})".format(prediction_types[p], aucs[-1]), lw=4,
                 markersize=3, ci=None, ax=ax, alpha=0.8)
    ax.tick_params(axis='both', which='major', labelsize=fontsize_3)
    plt.grid(True)
    plt.xlim([0, 1])
    plt.ylim([0.5, 1])
    plt.xlabel('Recall', fontsize=fontsize_2)
    plt.ylabel('Precision', fontsize=fontsize_2)
    plt.legend(loc="upper right", fontsize=fontsize_2)
    # plt.title('Performance over PathLinker Dataset', fontsize= fontsize)
    fig.set_size_inches(12, 8)
    plt.tight_layout()
    plt.savefig(path.join(output_file_path,'aucpr_curve_consensus_vs_all'))
sns.lineplot(x=[0, 1], y=[0.5, 0.5], dashes=True, color=model_colors['random'], label='Random(0.50)', lw=4, ci=None,
             ax=ax, alpha=0.8)
# fig, ax = plt.subplots()
# sns.histplot(d2d_positive_ranks, stat='density', alpha=0.3, bins=100, label='d2d positive score', ax=ax, color='darkblue')
# sns.histplot(positive_ranks, stat='density', alpha=0.3, bins=100, label='positive score', ax=ax, color='orangered')
# sns.histplot(ranks, stat='density', alpha=0.3, bins=100, label='positive score', ax=ax, color='green',element="step", fill=False)
# sns.histplot(d2d_negative_ranks, stat='density',  alpha=0.5, bins=500, label='negative_scores')
# sns.histplot(negative_ranks, stat='density',  alpha=0.5, bins=500, label='negative_scores')
