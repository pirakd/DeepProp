import matplotlib.pyplot as plt
import numpy as np
from os import path, makedirs
from utils import get_root_path, get_time
import json
import seaborn
from scripts.scripts_utils import model_colors
def load_results(folder_path):
    results_path = path.join(folder_path, 'results')

    with open(results_path, 'r') as f:
        results_dict = json.load(f)
    return results_dict

root_path = get_root_path()
output_folder = 'output'
output_file_path = path.join(get_root_path(), output_folder, path.basename(__file__).split('.')[0], get_time())
makedirs(output_file_path, exist_ok=True)

fontsize_1 = 24
fontsize_2 = 22
fontsize_3 = 22
fontsize_4 = 12
bar_width = 0.37
gap_width = 0.03

# d2d_folder_path = path.join(get_root_path(), 'input', 'other', '20_04_2022__15_24_22')
# deep_folder_path = path.join(get_root_path(), 'input', 'other', '20_04_2022__15_24_19')
#
# d2d_folder_path = path.join(get_root_path(), 'input', 'other', '20_04_2022__16_50_21')
# deep_folder_path = path.join(get_root_path(), 'input', 'other', '20_04_2022__16_50_17')

fraction_of_cancer_genes = 943/18800
d2d_folder_path = path.join(get_root_path(), 'input', 'other', '23_04_2022__22_11_50')
deep_folder_path = path.join(get_root_path(), 'input', 'other', '23_04_2022__22_08_47')

d2d_folder_path = path.join(get_root_path(), 'input', 'results', '23_06_2022__14_31_00')
deep_folder_path = path.join(get_root_path(), 'input', 'results', '26_06_2022__13_25_08')

d2d_results_dict = load_results(d2d_folder_path)
deep_results_dict = load_results(deep_folder_path)

test_gene_sets = list(d2d_results_dict['results']['ovary'].keys())
percentiles = [0.25, 0.5, 1, 5]


prediction_types = sorted(['breast', 'colon', 'AML', 'ovary'])

X = np.arange(4)*1.5

d2d_mean = []
deep_mean = []
undirected_mean = []

for percentile_idx, percentile in enumerate(percentiles):
    d2d_scores = []
    deep_scores = []
    undirected_scores = []

    for prediction in prediction_types:
        d2d_scores.append(d2d_results_dict['results'][prediction]['overall']['directed_network']['percentiles'][percentile_idx]/fraction_of_cancer_genes)
        deep_scores.append(deep_results_dict['results'][prediction]['overall']['directed_network']['percentiles'][percentile_idx]/fraction_of_cancer_genes)
        undirected_scores.append(deep_results_dict['results'][prediction]['overall']['undirected_network']['percentiles'][percentile_idx]/fraction_of_cancer_genes)

    d2d_mean.append(np.mean(d2d_scores))
    deep_mean.append(np.mean(deep_scores))
    undirected_mean.append(np.mean(undirected_scores))

    fig, ax = plt.subplots()
    ax.set_xticks(X+bar_width,prediction_types, fontsize=fontsize_2)
    ax.yaxis.set_tick_params(labelsize=fontsize_2)
    a = ax.bar(X + 0.00, d2d_scores,  width = bar_width, label='D2D', alpha= 1, color=model_colors['d2d'])
    b= ax.bar(X + bar_width+gap_width, deep_scores,  width = bar_width,label="D'OR", alpha=1, color=model_colors['deep'])

    c = ax.bar(X + ((bar_width+gap_width)*2), undirected_scores, width = bar_width, label='Unoriented', alpha=1, color=model_colors['unoriented'])
    ax.set_ylim((0, ax.get_ylim()[1]*1.15))
    ax.bar_label(b, padding=3, fmt='%.2f', fontsize=fontsize_2)
    ax.bar_label(a, padding=3, fmt='%.2f', fontsize=fontsize_2)
    ax.bar_label(c, padding=3, fmt='%.2f', fontsize=fontsize_2)
    plt.xlabel('Cancer Type', fontsize=fontsize_2)
    plt.ylabel('Fold Enrichment', fontsize=fontsize_2)
    # ax.set_title('Fold Enrichment of Top {}% Ranked Genes'.format(percentile), fontsize=fontsize_1)
    # ax.set_ylim()
    ax.legend(fontsize=fontsize_3, bbox_to_anchor=(0, 1), ncol=3, loc='upper left')
    fig.set_size_inches(12, 8)
    plt.tight_layout()
    plt.savefig(path.join(output_file_path, 'percentile_{}.jpg'.format(percentile)))
    plt.close()

fig, ax = plt.subplots()
ax.set_xticks(X+bar_width, percentiles, fontsize=fontsize_2)
ax.yaxis.set_tick_params(labelsize=fontsize_2)
a = ax.bar(X + 0.00, d2d_mean,  width = bar_width, label='D2D', alpha= 1, color=model_colors['d2d'])
b= ax.bar(X + bar_width+gap_width, deep_mean,  width = bar_width,label="D'OR", alpha=1, color=model_colors['deep'])
c = ax.bar(X + ((bar_width+gap_width)*2), undirected_mean, width = bar_width, label='Unoriented', alpha=1, color=model_colors['unoriented'])
ax.set_ylim((0, ax.get_ylim()[1]*1.15))
ax.bar_label(b, padding=3, fmt='%.2f', fontsize=fontsize_2)
ax.bar_label(a, padding=3, fmt='%.2f', fontsize=fontsize_2)
ax.bar_label(c, padding=3, fmt='%.2f', fontsize=fontsize_2)
plt.xlabel('% of top ranked genes', fontsize=fontsize_2)
plt.ylabel('Fold Enrichment', fontsize=fontsize_2)
# ax.set_title('Mean Fold Enrichment of Top Ranked Genes', fontsize=fontsize_1)
ax.set_ylim()
ax.legend(fontsize=fontsize_3, bbox_to_anchor=(0, 1), ncol=3, loc='upper left')
fig.set_size_inches(12, 8)
plt.tight_layout()
plt.savefig(path.join(output_file_path, 'percentile_mean.jpg'.format(percentile)))
plt.close()
# fig, ax = plt.subplots()
#
# seaborn.lineplot(x=np.arange(len(percentiles)), y=d2d_mean, marker='o', ax=ax, label='D2D', alpha=0.7)
# seaborn.lineplot(x=np.arange(len(percentiles)), y=deep_mean, ax=ax, label='DeepProp', alpha=0.7)
# seaborn.lineplot(x=np.arange(len(percentiles)), y=undirected_mean, ax=ax, label='Unoriented', alpha=0.7)
# ax.set_title('Fraction of cancer driver genes among top ranked genes', fontsize=fontsize_1)

a=1