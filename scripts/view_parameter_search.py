from os import listdir, path
import json
from utils import get_root_path
import numpy as np
import pandas as pd
# read all
# put in a pandas dataframe
# save in an numbers file ?

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
    fields_to_keep = ['best_epoch', 'best_auc', 'best_acc', 'best_eval_loss',
                      'classifier_layers', 'feature_extractor_layers', 'learning_rate', 'intermediate_loss_weight',
                      'exp_emb_size']

    root_path = get_root_path()
    result_list = read_results(path.join(root_path, 'output'))
    results_df = insert_to_data_frame(result_list, fields_to_keep)

    sorted = results_df.sort_values(by='best_auc', ascending=False)
    a=1

