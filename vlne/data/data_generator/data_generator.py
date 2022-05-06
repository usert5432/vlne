import math
import numpy as np

from vlndata.data_loader import vldata_dict_collate
from .idata_generator    import IDataGenerator

class DataGenerator(IDataGenerator):

    def __init__(
        self, dataset, input_groups, target_groups,
        batch_size = 1024,
        weights    = None,
    ):
        super().__init__(dataset, input_groups, target_groups)

        self._batch_size = batch_size
        self._weights    = { }

        weights = weights or {}

        for t in self.target_groups:
            if t in weights:
                self._weights[t] = dataset.df[weights[t]]
            else:
                self._weights[t] = np.ones(len(dataset))

    def get_data(self, indices):
        inputs  = {}
        targets = {}

        data_batch = [ self._dataset[index] for index in indices ]
        data_batch = vldata_dict_collate(data_batch, pad = 0)

        if len(data_batch) == 0:
            raise ValueError("Empty data batch extracted from dataset")

        inputs  = { k : data_batch[k] for k in self.input_groups }
        targets = { k : data_batch[k] for k in self.target_groups }
        weights = { k : w[indices] for (k, w) in self.weights.items() }

        return (inputs, targets, weights)

    def __len__(self):
        return math.ceil(len(self._dataset) / self._batch_size)

    @property
    def weights(self):
        return self._weights

    def __getitem__(self, index):
        start = index * self._batch_size
        end   = min((index + 1) * self._batch_size, len(self._dataset))

        return self.get_data(np.arange(start, end))

