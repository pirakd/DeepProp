import matplotlib.pyplot as plt
import numpy as np
from os import path, makedirs
from utils import get_root_path, get_time
import json
from scripts.scripts_utils import model_colors
import seaborn

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
bar_width = 0.37
X = np.arange(5)*1.3

gap_width = 0.02

deep_folder_path = path.join(get_root_path(), 'input', 'results', '17_08_2022__17_13_14_166')
d2d_folder_path = path.join(get_root_path(), 'input', 'results', '17_08_2022__17_13_48_821')


d2d_results_dict = load_results(d2d_folder_path)
deep_results_dict = load_results(deep_folder_path)

percentiles = [np.round(1-x,2)*100 for x in deep_results_dict['results']['left_unoriented']]

d2d_mean = []
deep_mean = []
undirected_mean = []

d2d_scores = d2d_results_dict['results']['fold_enrichment']
deep_scores = deep_results_dict['results']['fold_enrichment']
d2d = np.mean(d2d_scores)
deep = np.mean(deep_scores)


fig, ax = plt.subplots()
ax.set_xticks(X+bar_width, percentiles, fontsize=fontsize_2)
ax.yaxis.set_tick_params(labelsize=fontsize_2)
a = ax.bar(X + 0.00, d2d_scores,  width = bar_width, label='D2D', color=model_colors['d2d'],  alpha= 1)
b= ax.bar(X + bar_width+gap_width, deep_scores,  width = bar_width,label='D\'OR', color=model_colors['deep'], alpha=1)
ax.set_ylim((0, ax.get_ylim()[1]+0.15))
ax.bar_label(b, padding=3, fmt='%.2f', fontsize=fontsize_3)
ax.bar_label(a, padding=3, fmt='%.2f', fontsize=fontsize_3)
plt.xlabel('Percentile', fontsize=fontsize_2)
plt.ylabel('Fold Enrichment', fontsize=fontsize_2)
# ax.set_title('Fraction of disease genes among top ranked genes', fontsize=fontsize_1)
# ax.set_ylim((4, ax.get_ylim()[1] * 1.15))
ax.legend(fontsize=fontsize_3, bbox_to_anchor=(0, 1), ncol=3, loc='upper left')
plt.grid(axis='y')
fig.set_size_inches(12, 8)
plt.tight_layout()
plt.savefig(path.join(output_file_path, 'percentiles_complexes.jpg'))
plt.close()

# bar_width = 0.4
# X = np.arange(5)*1.5
# fig, ax = plt.subplots()
# custom_cycler = (cycler(color=['peru', 'royalblue', 'seagreen']))
# ax.set_prop_cycle(custom_cycler)
# ax.set_xticks(X+bar_width, percentiles, fontsize=fontsize_2)
# ax.yaxis.set_tick_params(labelsize=fontsize_2)
# a = ax.bar(X + 0.00, d2d_mean,  width = bar_width, label='D2D', alpha= 1)
# b= ax.bar(X + bar_width+gap_width, deep_mean,  width = bar_width,label='DeepProp', alpha=1)
# c = ax.bar(X + ((bar_width+gap_width)*2), undirected_mean, width = bar_width, label='Unoriented', alpha=1)
# ax.set_ylim((0, ax.get_ylim()[1]+0.06))
# ax.bar_label(b, padding=3, fmt='%.3f', fontsize=fontsize_3)
# ax.bar_label(a, padding=3, fmt='%.3f', fontsize=fontsize_3)
# ax.bar_label(c, padding=3, fmt='%.3f', fontsize=fontsize_3)
# plt.xlabel('% of top ranked genes', fontsize=fontsize_2)
# plt.ylabel('Fraction', fontsize=fontsize_2)
# ax.set_title('Mean fraction of cancer driver genes among top ranked genes', fontsize=fontsize_1)
# ax.set_ylim()
# ax.legend(fontsize=14, bbox_to_anchor=(0, 1), ncol=3, loc='upper left')
# plt.tight_layout()
# fig.set_size_inches(12, 8)
# plt.savefig(path.join(output_file_path, 'percentile_mean.jpg'))
# plt.close()
# fig, ax = plt.subplots()

# seaborn.lineplot(x=np.arange(len(percentiles)), y=d2d_mean, marker='o', ax=ax, label='D2D', alpha=0.7)
# seaborn.lineplot(x=np.arange(len(percentiles)), y=deep_mean, ax=ax, label='DeepProp', alpha=0.7)
# seaborn.lineplot(x=np.arange(len(percentiles)), y=undirected_mean, ax=ax, label='Unoriented', alpha=0.7)
# ax.set_title('Fraction of cancer driver genes among top ranked genes', fontsize=fontsize_1)
