from os import listdir, path, makedirs
import json
from utils import get_root_path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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

def plot_parameter_violin_plots(fields, results_df):
    df = results_df.copy()
    for field in fields:
        output_file = path.join(output_dir, field)
        fig, ax = plt.subplots()
        if not isinstance(df[field][0], list):
            order = df[field].unique().sort()
        else:
            df[field] = df[field].apply(lambda x: ''.join(str(x)))
            order = df[field].unique()
        plt.title('{} trained models'.format(len(df)))

        sns.stripplot(x=field, y="best_auc", data=df, order=order, alpha=0.1, ax=ax)
        sns.violinplot(x=field, y="best_auc", data=df, cut=0,
                            category_orders={field: order}, color='.8', alpha=1, ax=ax)
        minor_ticks_top = np.linspace(0.7, 1, 16)
        ax.set_yticks(minor_ticks_top)
        plt.grid(axis='y')
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()

def plot_quantiles(metric, quantiles=(95, 90, 80, 60, 50, 25, 10, 0)):
    output_file = path.join(output_dir, 'quantiles')

    bootstrap_quant = np.array([np.percentile(np.random.choice(metric, len(metric)),quantiles) for i in range(1000)])
    medians = np.median(bootstrap_quant, axis=0)
    quants_ci = np.percentile(bootstrap_quant,axis=0, q=[5, 95])
    quantiles = 100 - np.array(quantiles)

    plt.figure()
    plt.title('{} trained models'.format(len(metric)))

    plt.plot(quantiles, medians, marker='o', label='median')
    plt.fill_between(quantiles, y1=quants_ci[0, :], y2=quants_ci[1, :], alpha=0.5, label='95% CI')
    plt.grid()
    plt.xlabel('percentile of trained models by AUC')
    plt.ylabel('AUC')
    plt.legend()
    plt.savefig(output_file)


if __name__ == '__main__':
    script_name = path.basename(__file__).split('.')[0]
    output_dir = path.join(get_root_path(), 'output', script_name, get_time())
    makedirs(output_dir, exist_ok=True)
    fields_to_keep = ['best_epoch', 'best_auc', 'best_acc', 'best_val_loss',
                      'classifier_layers', 'feature_extractor_layers', 'learning_rate', 'intermediate_loss_weight',
                      'exp_emb_size','train_batch_size']
    fields_to_plot = ['classifier_layers', 'intermediate_loss_weight', 'feature_extractor_layers', 'exp_emb_size', 'learning_rate', 'train_batch_size']
    root_path = get_root_path()
    result_list = read_results(path.join(root_path, 'models_stats_50'))
    results_df = insert_to_data_frame(result_list, fields_to_keep)
    results_df.fillna(0, inplace=True)
    sorted = results_df.sort_values(by='best_auc', ascending=False)

    plot_parameter_violin_plots(fields_to_plot, results_df)

    metric = results_df['best_auc'].to_numpy()
    plot_quantiles(metric)

