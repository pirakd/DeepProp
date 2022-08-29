import matplotlib.pyplot as plt
import numpy as np
from os import path, makedirs
from utils import get_root_path, get_time
import seaborn as sns
from scripts.scripts_utils import model_colors
output_folder = 'output'
output_file_path = path.join(get_root_path(), output_folder, path.basename(__file__).split('.')[0], get_time())
makedirs(output_file_path, exist_ok=True)
linewidth = 6
fontsize= 40
fontsize_2 = 24
fontsize_3 = 22

n_exp = [5, 10, 20,  50, 100,  480]
d2d = [0.528, 0.562, 0.663,  0.721, 0.793, 0.820]
# d2d_2 = [0.725, 0.759, 0.823,  0.890,0.919, 0.967]
deep = [0.798, 0.807, 0.819,  0.879, 0.881, 0.891]

x = np.arange(len(n_exp))
y_grid = np.linspace(0.5, 1, 11)
#
# plt.title('PR-AUC VS Number of Experiments', fontsize=fontsize_1)
# plt.plot(n_exp, deep, marker='x', linewidth=linewidth, label='DeepProp')
# plt.plot(n_exp, d2d, marker='x' ,linewidth=linewidth, label='D2D')
# # plt.plot(n_exp, d2d_2, marker='x', linewidth=linewidth, label='D2D Deconstructed')
#

# plt.legend(fontsize=fontsize_2)
# plt.xlabel('#Experiments', fontsize=fontsize_2)
# plt.ylabel('PR-AUC', fontsize=fontsize_2)
# plt.savefig(path.join(output_folder,'fig'))

fig, ax = plt.subplots()

sns.lineplot(x=np.arange(len(n_exp)),
             y=deep,
             marker="o", markersize=linewidth * 2.5, color=model_colors['deep'],
             label="D'OR", lw=linewidth, ci=None, ax=ax)

sns.lineplot(x=np.arange(len(n_exp)), y=d2d, marker="o", markersize=linewidth*2.5,  color=model_colors['d2d'], label='D2D',
             lw=linewidth, ci=None, ax=ax)  # , markeredgecolor = 'dimgrey')
ax.tick_params(axis='both', which='major', labelsize=fontsize_3)

# plt.xlim([0, 1])
# plt.ylim([0.5, 1])
plt.xlabel('#patients', fontsize=fontsize_2)
plt.ylabel('AUCPR', fontsize=fontsize_2)
plt.legend(loc="lower right", fontsize=fontsize_2)
# plt.title('{}, {}, {} guiding sources, {} folds, {}'.format(' '.join(args['data']['directed_interactions_filename']),
# args['data']['sources_filename'].split('_')[-1],
# n_experiments, n_folds, source_type))
plt.title('AUCPR VS number of patients', fontsize=fontsize)
plt.xticks(np.arange(len(n_exp)), n_exp)
plt.yticks(y_grid)
plt.grid(True)


fig.set_size_inches(12, 8)
plt.tight_layout()
plt.savefig(path.join(output_file_path, 'fig'))

plt.close()