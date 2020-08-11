
import numpy as np


class TransductivePreprocessing:
    def __init__(self, nodes_per_class=20, n_validation=500, n_test=1000, seed=0):
        # self.seed = seed
        self.nodes_per_class = nodes_per_class
        self.n_validation = n_validation
        self.n_test = n_test
        np.random.seed(seed)

    @staticmethod
    def create_label_idx(label_idx, n_nodes):
        rel_label_idx = np.random.choice(label_idx, n_nodes, replace=False)
        return rel_label_idx

    @staticmethod
    def set_size_per_label(N, labels):
        node_per_class = N // len(labels)
        resid = N - node_per_class * len(labels)
        labels_split = {label: node_per_class for label in labels}
        labels_split[np.random.choice(labels)] += resid
        return labels_split

    def train_val_test_idx_split(self, y):
        labels = y.unique().numpy()
        idx_arr = np.arange(len(y))
        train_idx, val_idx, test_idx = [], [], []
        val_label_split = self.set_size_per_label(self.n_validation, labels)
        test_label_split = self.set_size_per_label(self.n_test, labels)
        for label in labels:
            label_idx = idx_arr[y == label]
            train_label_idx = self.create_label_idx(label_idx, self.nodes_per_class)
            train_idx += train_label_idx.tolist()
            rel_idx = np.setxor1d(label_idx, train_label_idx)
            val_label_idx = self.create_label_idx(rel_idx, val_label_split[label])
            val_idx += val_label_idx.tolist()
            rel_idx = np.setxor1d(rel_idx, val_label_idx)
            test_label_idx = self.create_label_idx(rel_idx, test_label_split[label])
            test_idx += test_label_idx.tolist()
        return train_idx, val_idx, test_idx





