"""Functions to save/load trained networks."""

import tqdm
import tensorflow
from vlne.args import Args

def load_model(savedir, compile = False):
    """Load trained network and its configuration saved under `savedir`"""
    # pylint: disable=redefined-builtin
    args = Args.load(savedir = savedir)

    model = tensorflow.keras.models.load_model(
        "%s/model.h5" % (savedir), compile = compile
    )

    return (args, model)

def precache(dgen, name = ''):
    dset = dgen.dataset
    pbar = tqdm.tqdm(dset, desc = f'Precaching {name}', total = len(dset))

    # pylint: disable=consider-using-enumerate
    for i in range(len(dset)):
        _ = dset[i]
        pbar.update()

    pbar.close()

