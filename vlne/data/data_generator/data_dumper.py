import os
import pickle

from .idata_decorator import IDataDecorator

class DataDumper(IDataDecorator):

    def __init__(self, dgen, outdir):
        super().__init__(dgen)

        self._outdir = outdir
        os.makedirs(outdir, exist_ok = True)

    def __getitem__(self, index):
        batch = self._dgen[index]
        path  = os.path.join(self._outdir, 'batch_%d.pkl' % (index))

        with open(path, 'wb') as f:
            pickle.dump(batch, f)

        return batch

