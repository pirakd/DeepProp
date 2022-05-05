import matplotlib.pyplot as plt
import numpy as np
from os import path, makedirs
from utils import get_root_path, get_time
import json
from cycler import cycler
import seaborn

root_path = get_root_path()
output_folder = 'output'
output_file_path = path.join(get_root_path(), output_folder, path.basename(__file__).split('.')[0], get_time())
makedirs(output_file_path, exist_ok=True)

deep_results_file_path = path.join(root_path, 'input', 'results', 'drug_kegg_0')
d2d_results_file_path = path.join(root_path, 'input', 'results', 'd2d_dummy')

with open(path.join(deep_results_file_path, 'results'), 'r' ) as f:
    deep_results_dict = json.load(f)['deep']

with open(path.join(deep_results_file_path, 'args'), 'r' ) as f:
    args = json.load(f)


with open(path.join(d2d_results_file_path, 'results'), 'r' ) as f:
    all_results = json.load(f)
    d2d_results = all_results['d2d']
    d2d_2_results = all_results['d2d_2']

n_folds = len(deep_results_dict['folds_stats'])
n_experiments = 'all'


for source_type in deep_results_dict['final'].keys():

    plt.plot([0, 1], [0.5, 0.5], '--', color=(0.8, 0.8, 0.8), label='random')
    plt.plot(deep_results_dict['final'][source_type]['recall'], deep_results_dict['final'][source_type]['precision'],
             'o--', label='DeepProp (%0.2f)' % deep_results_dict['final'][source_type]['auc'], lw=2,
             markersize=3)  # , markeredgecolor = 'dimgrey')
    plt.plot(d2d_results['final'][source_type]['recall'], d2d_results['final'][source_type]['precision'], 'o--', label='D2D (%0.2f)' % d2d_results['final'][source_type]['auc'], lw=2,
             markersize=3)  # , markeredgecolor = 'dimgrey')
    plt.plot(d2d_2_results['final'][source_type]['recall'], d2d_2_results['final'][source_type]['precision'], 'o--', label='D2D Deconstructed (%0.2f)' % d2d_2_results['final'][source_type]['auc'], lw=2,
             markersize=3)  # , markeredgecolor = 'dimgrey')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="lower right")
    plt.title('{}, {}, {} sources, {} folds, {}'.format(' '.join(args['data']['directed_interactions_filename']),
                                                        args['data']['sources_filename'].split('_')[-1],
                                                        n_experiments, n_folds, source_type))
    plt.savefig(path.join(output_file_path, 'auc_curve_{}'.format(source_type)))

    plt.close()