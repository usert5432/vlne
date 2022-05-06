from .idata_generator import IDataGenerator

class IDataDecorator(IDataGenerator):

    def __init__(self, dgen):
        super().__init__(dgen.dataset, dgen.input_groups, dgen.target_groups)
        self._dgen = dgen

    def __len__(self):
        return len(self._dgen)

    def __getitem__(self, index):
        return self._dgen[index]

    @property
    def dataset(self):
        return self._dgen.dataset

    @property
    def weights(self):
        return self._dgen.weights

