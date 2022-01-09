from torch.utils.data import Dataset
import numpy as np
import torch
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


def train_test_split(n_samples, train_test_ratio):
    is_train = np.random.binomial(1, train_test_ratio, n_samples)
    train_indexes = np.nonzero(is_train)[0]
    test_indexes = np.nonzero(1 - is_train)[0]

    return train_indexes, test_indexes