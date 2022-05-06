from tensorflow.keras.utils import Sequence
from .idata_decorator import IDataDecorator

class KerasSequence(IDataDecorator, Sequence):

    def __init__(self, dgen):
        IDataDecorator.__init__(self, dgen)

    def __len__(self):
        return len(self._dgen)

    def __getitem__(self, index):
        return self._dgen[index]

