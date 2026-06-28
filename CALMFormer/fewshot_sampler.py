import numpy as np
import torch


class FewShotEpisodeSampler:
    """Sample N-way K-shot episodes from a single dataset."""

    def __init__(self, dataset, n_way=5, k_shot=1, q_query=15, seed=2027):
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.rng = np.random.default_rng(seed)

        labels = np.asarray(dataset.Y)
        self.class_to_indices = {
            int(cls): np.where(labels == cls)[0]
            for cls in np.unique(labels)
        }
        self.classes = sorted(self.class_to_indices.keys())
        self._check_enough_samples()

    def _check_enough_samples(self):
        if len(self.classes) < self.n_way:
            raise ValueError(f"n_way={self.n_way} exceeds number of classes={len(self.classes)}")

        required = self.k_shot + self.q_query
        too_small = {
            cls: len(indices)
            for cls, indices in self.class_to_indices.items()
            if len(indices) < required
        }
        if too_small:
            raise ValueError(f"Some classes have fewer than {required} samples: {too_small}")

    def sample_episode(self):
        selected_classes = self.rng.choice(self.classes, size=self.n_way, replace=False)
        support_x, support_y = [], []
        query_x, query_y = [], []

        for new_label, cls in enumerate(selected_classes):
            indices = self.class_to_indices[int(cls)]
            chosen = self.rng.choice(indices, size=self.k_shot + self.q_query, replace=False)

            for idx in chosen[:self.k_shot]:
                x, _ = self.dataset[int(idx)]
                support_x.append(x)
                support_y.append(new_label)

            for idx in chosen[self.k_shot:]:
                x, _ = self.dataset[int(idx)]
                query_x.append(x)
                query_y.append(new_label)

        return (
            torch.stack(support_x, dim=0),
            torch.tensor(support_y, dtype=torch.long),
            torch.stack(query_x, dim=0),
            torch.tensor(query_y, dtype=torch.long),
            selected_classes.tolist(),
        )


class CrossSplitFewShotEpisodeSampler:
    """Sample support and query examples from separate dataset splits."""

    def __init__(self, support_dataset, query_dataset, n_way=5, k_shot=1, q_query=15, seed=9001):
        self.support_dataset = support_dataset
        self.query_dataset = query_dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.rng = np.random.default_rng(seed)

        self.support_class_to_indices = self._build_class_index(support_dataset)
        self.query_class_to_indices = self._build_class_index(query_dataset)
        self.classes = sorted(set(self.support_class_to_indices) & set(self.query_class_to_indices))
        self._check_enough_samples()

    @staticmethod
    def _build_class_index(dataset):
        labels = np.asarray(dataset.Y)
        return {
            int(cls): np.where(labels == cls)[0]
            for cls in np.unique(labels)
        }

    def _check_enough_samples(self):
        if len(self.classes) < self.n_way:
            raise ValueError(f"n_way={self.n_way} exceeds number of shared classes={len(self.classes)}")

        support_too_small = {
            cls: len(self.support_class_to_indices[cls])
            for cls in self.classes
            if len(self.support_class_to_indices[cls]) < self.k_shot
        }
        query_too_small = {
            cls: len(self.query_class_to_indices[cls])
            for cls in self.classes
            if len(self.query_class_to_indices[cls]) < self.q_query
        }
        if support_too_small:
            raise ValueError(f"Some support classes have fewer than {self.k_shot} samples: {support_too_small}")
        if query_too_small:
            raise ValueError(f"Some query classes have fewer than {self.q_query} samples: {query_too_small}")

    def sample_episode(self):
        selected_classes = self.rng.choice(self.classes, size=self.n_way, replace=False)
        support_x, support_y = [], []
        query_x, query_y = [], []

        for new_label, cls in enumerate(selected_classes):
            support_indices = self.rng.choice(
                self.support_class_to_indices[int(cls)],
                size=self.k_shot,
                replace=False,
            )
            query_indices = self.rng.choice(
                self.query_class_to_indices[int(cls)],
                size=self.q_query,
                replace=False,
            )

            for idx in support_indices:
                x, _ = self.support_dataset[int(idx)]
                support_x.append(x)
                support_y.append(new_label)

            for idx in query_indices:
                x, _ = self.query_dataset[int(idx)]
                query_x.append(x)
                query_y.append(new_label)

        return (
            torch.stack(support_x, dim=0),
            torch.tensor(support_y, dtype=torch.long),
            torch.stack(query_x, dim=0),
            torch.tensor(query_y, dtype=torch.long),
            selected_classes.tolist(),
        )
