from os import listdir, path, makedirs
import json
from utils import get_root_path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import get_time
from functools import reduce
import operator
import datetime

model_args = ['classifier_layers', 'feature_extractor_layers', 'exp_emb_size', 'feature_extractor_dropout',
              'classifier_dropout', 'pair_degree_feature']
train_args = ['intermediate_loss_weight', 'learning_rate', 'train_batch_size']
data_args = ['normalization_method']

model_args = [['args','model',x] for x in model_args]
train_args = [['args','train', x] for x in train_args]
data_args = [['args', 'data', x] for x in data_args]
datasets = ['overall','KPI', 'STKE', 'E3', 'EGFR' ]
results = [['results','val_stats', x ,'mean_auc'] for x in datasets]
fields_to_keep = [x for xx in [model_args, train_args, data_args, results] for x in xx]
fields_to_keep.append(['results','train_stats', 'best_epoch'])

field_to_plot_names = (['#'.join(x) for xx in [model_args, train_args, data_args] for x in xx])
metric = results[0]
metric_name = '#'.join(metric)


def get_dict_item(key, dictionary):
    return reduce(operator.getitem, key, dictionary)

def read_results(results_folder, start_date=None):
    result_files = listdir(results_folder)
    if start_date:
        start_date = datetime.datetime.strptime(start_date, '%d_%m_%Y__%H_%M_%S')
        result_files = [x for x in result_files if x != '.DS_Store' and datetime.datetime.strptime(x, '%d_%m_%Y__%H_%M_%S')>start_date]
    results_list = []
    for file in result_files:
        if file != '.DS_Store':
            if len(listdir(path.join(results_folder,path.join(file)))) == 2:
                with open(path.join(results_folder,path.join(file,'args')), 'r') as f:
                    args_dict = json.load(f)
                with open(path.join(results_folder,path.join(file,'results')), 'r') as f:
                    results_dict = json.load(f)
                results_list.append(dict())
                # results_list[-1]['args'] = pd.json_normalize(args_dict, sep='@').to_dict(orient='records')[0]
                # results_list[-1]['results'] = pd.json_normalize(results_dict, sep='@').to_dict(orient='records')[0]
                results_list[-1]['args'] = args_dict
                results_list[-1]['results'] = results_dict
    return results_list


def insert_to_data_frame(result_list, fields_to_keep):
    filtered_results = list()
    # arg_to_idx = {arg:idx for idx, arg in enumerate(fields_to_keep)}
    # idx_to_args = {xx:x for x,xx in arg_to_idx.items()}

    arg_to_idx = {'#'.join(arg):idx for idx, arg in enumerate(fields_to_keep)}

    for i in range(len(result_list)):
        filtered_results.append([None]*len(fields_to_keep))
        for arg in fields_to_keep:
            filtered_results[-1][arg_to_idx['#'.join(arg)]] = get_dict_item(arg, result_list[i])

    filtered_results = pd.DataFrame(filtered_results, columns=list(arg_to_idx.keys()))
    return filtered_results

def plot_parameter_violin_plots(results_df, fields, metric_name):
    df = results_df.copy()
    if not fields:
        fields = results_df.columns
    for field in fields:
        output_file = path.join(output_dir, field)
        fig, ax = plt.subplots()
        if not isinstance(df[field][0], list):
            order = df[field].unique().sort()
        else:
            df[field] = df[field].apply(lambda x: ''.join(str(x)))
            order = df[field].unique()
        plt.title('{} trained models'.format(len(df)))

        sns.stripplot(x=field, y=metric_name, data=df, order=order, alpha=0.15, ax=ax)
        sns.violinplot(x=field, y=metric_name, data=df, cut=0,
                            category_orders={field: order}, color='.8', alpha=1, ax=ax)
        # plt.ylim([0.89, 1])
        # minor_ticks_top = np.linspace(0.89, 1, 16)
        # ax.set_yticks(minor_ticks_top)
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

    root_path = get_root_path()
    result_list = read_results(path.join(root_path,'output', 'parameter_search'), '15_04_2022__01_01_01')
    results_df = insert_to_data_frame(result_list, fields_to_keep)
    results_df.fillna(0, inplace=True)
    sorted = results_df.sort_values(by=metric_name, ascending=False)
    # sorted = sorted[sorted['n_experiments'] == 551].reset_index()
    # sorted = sorted[sorted['best_auc'] > 0.9]
    plot_parameter_violin_plots(sorted, field_to_plot_names, metric_name)

    metric = sorted[metric_name].to_numpy()
    plot_quantiles(metric)

