from torch.utils.data import Dataset
import numpy as np
import torch
from sklearn.preprocessing import  PowerTransformer

class ClassifierDataset(Dataset):
    def __init__(self, source_features, terminal_features):
        self.n_experiments = len(source_features)
        self.n_directed_edges = len(source_features[0])
        self.dataset = self.generate_dataset(source_features, terminal_features)

    def __len__(self):
        return self.dataset['source'].shape[0]

    def __getitem__(self, idx):
        if type(idx) == torch.Tensor:
            idx = idx.item()

        return self.dataset['source'][idx], self.dataset['terminal'][idx], self.dataset['labels'][idx]

    def generate_dataset(self, source_features, terminal_features):
        source_samples = []
        terminal_samples = []
        batch_labels = []
        for i in range(self.n_directed_edges * 2):
            label = 1

            source_experiments_samples = []
            terminal_experiments_samples = []

            for e in range(self.n_experiments):

                exp_source_score = source_features[e][np.mod(i, self.n_directed_edges), ...]
                exp_terminal_score = terminal_features[e][np.mod(i, self.n_directed_edges), ...]
                if i >= self.n_directed_edges:
                    exp_source_score = np.flip(exp_source_score, axis=1)
                    exp_terminal_score = np.flip(exp_terminal_score, axis=1)
                    label = 0

                source_experiments_samples.append(exp_source_score)
                terminal_experiments_samples.append(exp_terminal_score)
            source_samples.append(source_experiments_samples)
            terminal_samples.append(terminal_experiments_samples)
            batch_labels.append(label)

        largest_source_set = np.max([source_samples[0][x].shape[0] for x in range(self.n_experiments)])
        largest_terminal_set = np.max([terminal_samples[0][x].shape[0] for x in range(self.n_experiments)])

        for sample_idx in range(len(source_samples)):
            for exp_idx in range(self.n_experiments):
                source_samples[sample_idx][exp_idx] = np.pad(source_samples[sample_idx][exp_idx], [(0, largest_source_set-source_samples[sample_idx][exp_idx].shape[0]), (0, 0)] )
                terminal_samples[sample_idx][exp_idx] = np.pad(terminal_samples[sample_idx][exp_idx], [(0, largest_terminal_set-terminal_samples[sample_idx][exp_idx].shape[0]), (0, 0)] )

        source_samples = np.array(source_samples)
        terminal_samples = np.array(terminal_samples)

        return {'source': source_samples, 'terminal': terminal_samples, 'labels': batch_labels}


class LightDataset(Dataset):
    def __init__(self, row_id_to_idx, col_id_to_idx, propagation_scores, directed_pairs_list,
                 sources, terminals, normalization_method, samples_normalization_constants,
                 degree_feature_normalization_constants=None,
                 pairs_source_type=None, id_to_degree=None, train=True):
        self.row_id_to_idx = row_id_to_idx
        self.col_id_to_idx = col_id_to_idx
        self.col_idx_to_id = {xx: x for x, xx in self.col_id_to_idx.items()}
        self.propagation_scores = propagation_scores
        self.source_indexes = self.get_experiment_indexes(sources)
        self.terminal_indexes = self.get_experiment_indexes(terminals)
        self.pairs_indexes = [(self.col_id_to_idx[pair[0]], self.col_id_to_idx[pair[1]]) for pair in directed_pairs_list]
        self.longest_source = np.max([len(source) for source in sources.values()])
        self.longest_terminal = np.max([len(terminal) for terminal in terminals.values()])
        normalizer = self.get_normalization_method(normalization_method)
        self.normalizer = normalizer(samples_normalization_constants)
        self.pairs_source_type = pairs_source_type
        self.idx_to_degree = {self.col_id_to_idx[id]: degree for id, degree in id_to_degree.items() if id in self.col_id_to_idx}
        self.degree_normalizer = self.get_degree_normalizar(degree_feature_normalization_constants)
        self.train = train
    def __len__(self):
        return len(self.pairs_indexes) * 2

    def __getitem__(self, idx):
        if type(idx) == torch.Tensor:
            idx = idx.item()
        neg_flag = idx > len(self.pairs_indexes)
        idx = np.mod(idx, len(self.pairs_indexes))
        label = 0 if neg_flag else 1
        pair = self.pairs_indexes[idx]
        if neg_flag:
            pair = (pair[1], pair[0])

        from_degree = self.degree_normalizer(self.idx_to_degree[pair[0]])
        to_degree =  self.degree_normalizer(self.idx_to_degree[pair[1]])

        pair_source_type = self.pairs_source_type[idx] if self.pairs_source_type is not None else None
        source_sample = np.zeros((len(self.source_indexes), self.longest_source, 2))
        terminal_sample = np.zeros((len(self.source_indexes), self.longest_terminal, 2))

        for exp_idx in range(len(self.source_indexes)):
            source_sample[exp_idx, :len(self.source_indexes[exp_idx]), :] =\
                self.normalizer(self.propagation_scores[:, pair][self.source_indexes[exp_idx], :])
            terminal_sample[exp_idx, :len(self.terminal_indexes[exp_idx]), :] =\
                self.normalizer(self.propagation_scores[:, pair][self.terminal_indexes[exp_idx], :])

        return source_sample, terminal_sample, label, pair, pair_source_type, np.array([from_degree, to_degree])

    def get_experiment_indexes(self, experiment_id_sets):
        experiment_indexes = []
        for set in experiment_id_sets.values():
            experiment_indexes.append([self.row_id_to_idx[id] for id in set])

        return experiment_indexes

    @staticmethod
    def get_normalization_method(normalization_method):
        if normalization_method == 'standard':
            return StandardNormalizer
        elif normalization_method == 'power':
            return PowerNormalizer
        else:
            assert 0, '{} is not a valid normalization method name'.format(normalization_method)

    def get_degree_normalizar(self, normalization_constants):
        if normalization_constants is None:
            all_degrees = []
            for pair in self.pairs_indexes:
                all_degrees.append(self.idx_to_degree[pair[0]])
                all_degrees.append(self.idx_to_degree[pair[1]])
            all_degrees = np.array(all_degrees)[:, np.newaxis]
            pt = PowerTransformer(method='box-cox', standardize=False)
            transformed = pt.fit_transform(all_degrees)
            mean = np.mean(transformed)
            std = np.std(transformed)
            lmbda = pt.lambdas_[0]
        else:
            mean = normalization_constants['mean']
            std = normalization_constants['std']
            lmbda = normalization_constants['lmbda']
        return PowerNormalizer({'lmbda': lmbda, 'mean': mean, 'std': std})

class StandardNormalizer():
    def __init__(self, normalization_constants):
        self.dataset_mean = normalization_constants['mean']
        self.dataset_std = normalization_constants['std']

    def __call__(self, x, *args, **kwargs):
        return (x-self.dataset_mean) / self.dataset_std


class PowerNormalizer:
    """
    yeo_johnson transform
    """
    def __init__(self, normalization_constants):
        from scipy.stats import yeojohnson
        from functools import partial
        self.transformer = partial(yeojohnson, lmbda=normalization_constants['lmbda'])
        self.lmbda = normalization_constants['lmbda']
        self.mean = normalization_constants['mean']
        self.std = normalization_constants['std']

    def __call__(self, x, *args, **kwargs):
        return (self.transformer(x) - self.mean) / self.std