from os import path
import sys
sys.path.append(path.dirname(path.dirname(path.realpath(__file__))))
from os import makedirs
from deep_learning.data_loaders import LightDataset
from deep_learning.trainer import ClassifierTrainer
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from utils import read_data, generate_raw_propagation_scores, load_model, log_results, get_time, \
    get_root_path, save_propagation_score, load_pickle, train_test_split, get_normalization_constants, get_loss_function
import torch
import numpy as np
import json

device = torch.device("cuda".format() if torch.cuda.is_available() else "cpu")
output_folder = 'output'
output_file_path = path.join(get_root_path(), output_folder, path.basename(__file__).split('.')[0], get_time())
makedirs(output_file_path, exist_ok=True)

root_path = get_root_path()
model_path = path.join(root_path, 'output', 'main' , '22_03_2022__17_08_36', 'model')
model_args_path = path.join(root_path,'output', 'main', '22_03_2022__17_08_36', 'args')
with open(model_args_path ,'r') as f:
    args = json.load(f)
train_dataset = args['data']['directed_interactions_filename']
args['data']['directed_interactions_filename'] = 'STKE_dataset'
args['train']['train_val_test_split'] = [0, 0, 1]
device = torch.device("cuda".format() if torch.cuda.is_available() else "cpu")
cmd_args = [int(arg) for arg in sys.argv[1:]]
if len(cmd_args) == 1:
    device = torch.device("cuda:{}".format(cmd_args[0]) if torch.cuda.is_available() else "cpu")

rng = np.random.RandomState(args['data']['random_seed'])
model = load_model(model_path, args).to(device)

# data read
network, directed_interactions, sources, terminals =\
    read_data(args['data']['network_filename'], args['data']['directed_interactions_filename'],
              args['data']['sources_filename'], args['data']['terminals_filename'],
              args['data']['n_experiments'], args['data']['max_set_size'], rng)


n_experiments = len(sources.keys())
directed_interactions_pairs_list = np.array(directed_interactions.index)
genes_ids_to_keep = sorted(list(set([x for pair in directed_interactions_pairs_list for x in pair])))

if args['data']['load_prop_scores']:
    scores_file_path = path.join(root_path, 'input', 'propagation_scores', args['data']['prop_scores_filename'])
    scores_dict = load_pickle(scores_file_path)
    propagation_scores = scores_dict['propagation_scores']
    row_id_to_idx, col_id_to_idx = scores_dict['row_id_to_idx'], scores_dict['col_id_to_idx']
    normalization_constants_dict = scores_dict['normalization_constants']
    sources_indexes = [[row_id_to_idx[id] for id in set] for set in sources.values()]
    terminals_indexes = [[row_id_to_idx[id] for id in set] for set in terminals.values()]
    pairs_indexes = [(col_id_to_idx[pair[0]], col_id_to_idx[pair[1]]) for pair in directed_interactions_pairs_list]
    assert scores_dict['data_args']['random_seed'] == args['data']['random_seed'], 'random seed of loaded data does not much current one'
else:
    propagation_scores, row_id_to_idx, col_id_to_idx = generate_raw_propagation_scores(network, sources, terminals, genes_ids_to_keep,
                                                                  args['propagation']['alpha'], args['propagation']['n_iterations'],
                                                                  args['propagation']['eps'])
    sources_indexes = [[row_id_to_idx[id] for id in set] for set in sources.values()]
    terminals_indexes = [[row_id_to_idx[id] for id in set] for set in terminals.values()]
    pairs_indexes = [(col_id_to_idx[pair[0]], col_id_to_idx[pair[1]]) for pair in directed_interactions_pairs_list]
    normalization_constants_dict = get_normalization_constants(pairs_indexes, sources_indexes, terminals_indexes,
                                                               propagation_scores)
    if args['data']['save_prop_scores']:
        save_propagation_score(propagation_scores, normalization_constants_dict, row_id_to_idx, col_id_to_idx,
                               args['propagation'], args['data'], 'balanced_kpi_prop_scores')


train_indexes, val_indexes, test_indexes = train_test_split(len(directed_interactions_pairs_list), args['train']['train_val_test_split'],
                                                            random_state=rng)
test_dataset = LightDataset(row_id_to_idx, col_id_to_idx, propagation_scores, directed_interactions_pairs_list[test_indexes], sources, terminals, normalization_constants_dict)
test_loader = DataLoader(test_dataset, batch_size=args['train']['test_batch_size'], shuffle=False, pin_memory=True, )
optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args['train']['learning_rate'])
trainer = ClassifierTrainer(args['train']['n_epochs'], criteria=nn.CrossEntropyLoss(), intermediate_criteria=None,
                            intermediate_loss_weight=args['train']['intermediate_loss_weight'],
                            optimizer=None, eval_metric=None, eval_interval=args['train']['eval_interval'], device='cpu')

test_loss, test_intermediate_loss, test_classifier_loss, test_acc, test_auc, precision, recall = \
    trainer.eval(model, test_loader)
print('Test PR-AUC: {:.2f}'.format(test_auc))
test_stats = {'test_loss': test_loss, 'best_acc': test_acc, 'best_auc': test_auc,
              'test_intermediate_loss': test_intermediate_loss, 'test_classifier_loss': test_classifier_loss}


results_dict = {'test_stats': test_stats, 'n_experiments': n_experiments, 'train_dataset':train_dataset, 'test_dataset':args['data']['directed_interactions_filename'],
              'model_path': model_path}
log_results(output_file_path,  args, results_dict)
