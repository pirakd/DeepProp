import matplotlib.pyplot as plt
import numpy as np
from os import path, makedirs
from utils import get_root_path, get_time
import json
from cycler import cycler
import seaborn as sns
from scripts.scripts_utils import model_colors
linewidth = 6
fontsize= 40
fontsize_2 = 24
fontsize_3 = 22
root_path = get_root_path()
output_folder = 'output'
output_file_path = path.join(get_root_path(), output_folder, path.basename(__file__).split('.')[0], get_time())
makedirs(output_file_path, exist_ok=True)
d2d_result_files={'drug':'22_06_2022__10_53_10',
                  'colon':'22_06_2022__10_52_43',
                  'AML':'22_06_2022__10_52_11',
                  'ovary': '22_06_2022__10_52_23',
                  'breast':'22_06_2022__10_51_55'}
deep_result_files={'drug':'22_06_2022__11_32_34',
                   'colon':'21_06_2022__17_55_25',
                   'AML':'21_06_2022__17_56_10',
                   'ovary': '22_06_2022__19_00_53',
                   'breast':'25_06_2022__12_29_15'}
vinayagam_result_files={'drug':'21_06_2022__15_31_40',
                        'colon':'21_06_2022__15_31_40',
                        'AML':'21_06_2022__15_31_40',
                        'ovary':'21_06_2022__15_31_40',
                        'breast':'21_06_2022__15_31_40'}

guiding_sources_to_plot = ['drug', 'AML', 'colon', 'ovary', 'breast']

for i, guiding_source in enumerate(guiding_sources_to_plot):
    deep_results_file_path = path.join(root_path, 'input', 'results', deep_result_files[guiding_source])
    d2d_results_file_path = path.join(root_path, 'input', 'results', d2d_result_files[guiding_source])
    vinayagam_results_file_path = path.join(root_path, 'input', 'results', vinayagam_result_files[guiding_source])

    with open(path.join(deep_results_file_path, 'results'), 'r' ) as f:
        deep_results_dict = json.load(f)['deep']

    with open(path.join(deep_results_file_path, 'args'), 'r' ) as f:
        args = json.load(f)


    with open(path.join(d2d_results_file_path, 'results'), 'r' ) as f:
        all_results = json.load(f)
        d2d_results = all_results['d2d']

    with open(path.join(vinayagam_results_file_path, 'results'), 'r' ) as f:
        all_results = json.load(f)
        vinayagam_results = all_results['vinayagam']


    n_folds = len(deep_results_dict['folds_stats'])
    n_experiments = 'all'

    # for source_type in deep_results_dict['final'].keys():
    for source_type in ['overall']:
        fig,ax = plt.subplots()
        sns.lineplot(x=[0, 1], y=[0.5, 0.5], linestyle="dashed", color=model_colors['random'], label='Random (0.50)', ci=None, ax=ax, lw=linewidth)
        sns.lineplot(x=deep_results_dict['final'][source_type]['recall'], y=deep_results_dict['final'][source_type]['precision'],
                 markers='o--', color=model_colors['deep'], label="D'OR (%0.2f)" % deep_results_dict['final'][source_type]['auc'], lw=linewidth,
                 markersize=3, ci=None, ax=ax)
        sns.lineplot(x=vinayagam_results['final'][source_type]['recall'],
                 y=vinayagam_results['final'][source_type]['precision'], markers='o--', color=model_colors['vinayagam'],
                 label='Vinayagam (%0.2f)' % vinayagam_results['final'][source_type]['auc'], lw=linewidth, markersize=3, ci=None, ax=ax)
        sns.lineplot(x=d2d_results['final'][source_type]['recall'], y=d2d_results['final'][source_type]['precision'],
                     markers='o--',  color=model_colors['d2d'], label='D2D(%0.2f)' % d2d_results['final'][source_type]['auc'], lw=linewidth,
                 markersize=3, ci=None, ax=ax)  # , markeredgecolor = 'dimgrey')
        ax.tick_params(axis='both', which='major', labelsize=fontsize_3)

        plt.xlim([0, 1])
        plt.ylim([0.5, 1])
        plt.xlabel('Recall', fontsize=fontsize_2)
        plt.ylabel('Precision', fontsize=fontsize_2)
        plt.legend(loc="lower left", fontsize=fontsize_2)
        # plt.title('{}, {}, {} guiding sources, {} folds, {}'.format(' '.join(args['data']['directed_interactions_filename']),
                                                            # args['data']['sources_filename'].split('_')[-1],
                                                            # n_experiments, n_folds, source_type))
        plt.title('{}'.format(guiding_source), fontsize=fontsize)
        plt.grid(True)
        fig.set_size_inches(12, 8)
        plt.tight_layout()
        plt.savefig(path.join(output_file_path, '{}_auc_curve_{}'.format(args['data']['sources_filename'], source_type)))

        plt.close()