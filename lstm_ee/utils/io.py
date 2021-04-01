"""Functions to save/load trained networks."""

import tensorflow
from lstm_ee.args import Args

def load_model(savedir, compile = False):
    """Load trained network and its configuration saved under `savedir`"""
    # pylint: disable=redefined-builtin
    args = Args.load(savedir = savedir)

    model = tensorflow.keras.models.load_model(
        "%s/model.h5" % (savedir), compile = compile
    )

    return (args, model)

