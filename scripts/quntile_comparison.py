from os import listdir, path, makedirs
import json
from utils import get_root_path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from utils import get_time

def read_results(results_folder):
    result_files = listdir(results_folder)

    results_list = []
    for file in result_files:
        with open(path.join(results_folder, file), 'r') as f:
            dictionary = json.load(f)
            results_list.append(pd.json_normalize(dictionary, sep='@').to_dict(orient='records'))
    return results_list


def insert_to_data_frame(result_list, fields_to_keep):
    filtered_results = list()
    arg_to_idx = {arg:idx for idx, arg in enumerate(fields_to_keep)}
    idx_to_args = {xx:x for x,xx in arg_to_idx.items()}
    for i in range(len(result_list)):
        filtered_results.append([None]*len(fields_to_keep))
        for arg_class in result_list[i]:
            for arg_name, value in arg_class.items():
                arg_name = arg_name.split('@')[-1]
                if arg_name in arg_to_idx:
                    filtered_results[-1][arg_to_idx[arg_name]] = value

    filtered_results = pd.DataFrame(filtered_results, columns=fields_to_keep)
    return filtered_results


if __name__ == '__main__':
    metric_name = 'best_auc'
    script_name = path.basename(__file__).split('.')[0]
    output_dir = path.join(get_root_path(), 'output', script_name, get_time())
    output_file = path.join(output_dir, 'quantiles')
    makedirs(output_dir, exist_ok=True)

    distinguish_by_field = 'n_experiments'
    fields_to_keep = ['best_auc', 'n_experiments']
    root_path = get_root_path()
    result_list = read_results(path.join(root_path, 'models_stats'))
    results_df = insert_to_data_frame(result_list, fields_to_keep)
    results_df.fillna(0, inplace=True)
    sorted = results_df.sort_values(by='best_auc', ascending=False)

    fig, ax = plt.subplots()
    # plt.title('{} trained models'.format(len(metric)))
    plt.grid()
    plt.xlabel('Percentile of trained models by AUC')
    plt.ylabel('AUC')

    unique_values = results_df[distinguish_by_field].unique()
    for value in unique_values:
        quantiles = (95, 90, 80, 60, 50, 25, 10, 0)
        metric = results_df[results_df[distinguish_by_field] == value][metric_name]
        bootstrap_quant = np.array([np.percentile(np.random.choice(metric, len(metric)),quantiles) for i in range(1000)])
        medians = np.median(bootstrap_quant, axis=0)
        quants_ci = np.percentile(bootstrap_quant,axis=0, q=[5, 95])
        quantiles = 100 - np.array(quantiles)
        ax.plot(quantiles, medians, marker='o', label='{}: {}'.format(distinguish_by_field, value))
        ax.fill_between(quantiles, y1=quants_ci[0, :], y2=quants_ci[1, :], alpha=0.5)

    handles, labels = ax.get_legend_handles_labels()
    patch = mpatches.Patch(color='grey', label='95% CI', alpha=0.5)
    handles.append(patch)
    plt.legend(handles=handles, loc='lower left')
    plt.savefig(output_file)