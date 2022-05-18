"""
Functions to train `vlne` model.
"""

import logging
import numpy as np

from vlne.args       import Args
from vlne.args.funcs import update_kwargs
from vlne.data       import load_data
from vlne.utils.io   import precache
from .setup       import (
    get_optimizer, get_default_callbacks, get_keras_concurrency_kwargs,
    select_model, limit_tf_memory_growth
)

LOGGER = logging.getLogger('vlne.train')

def return_training_stats(train_log, savedir):
    """Return a dict with a summary of training results.

    Parameters
    ----------
    train_log : keras.History
        Training history returned by `keras.model.fit`
    savedir : str
        Directory where trained model is saved.

    Return
    ------
    dict
        Dictionary with training summary.
    """

    best_idx = np.argmin(train_log.history['val_loss'])

    result = {
        'loss'    : (
            train_log.history['val_target_total_loss'][best_idx]
          + train_log.history['val_target_primary_loss'][best_idx]
        ),
        'status'  : 0,
        'time'    : train_log.history['train_time'][-1],
        'epochs'  : len(train_log.history['val_loss']),
        'savedir' : savedir
    }

    return result

def create_and_train_model(extra_kwargs = None, **args_dict):
    """Creates and trains `keras` model specified by arguments.

    Parameters
    ----------
    args : Args or None, optional
        Specification of the model and training setup
        If None, then the model and training specification will be first
        constructed from `kwargs` and `extra_kwargs`
    extra_kwargs : dict or None, optional
        Extra kwargs that will be passed to the `Args` constructor.
    kwargs : dict
        Parameters that will be passed to the `Args` constructor if `args` is
        None.

    Return
    ------
    dict
        Dictionary with training summary returned by `return_training_stats`.

    See Also
    --------
    vlne.args.Args
    return_training_stats
    """

    limit_tf_memory_growth()

    if extra_kwargs is not None:
        update_kwargs(args_dict, extra_kwargs)

    args = Args.from_args_dict(**args_dict)

    LOGGER.info(
        "Starting training with parameters:\n%s", args.config.pprint()
    )

    LOGGER.info("Loading data...")
    dgen_train, dgen_test = load_data(args, [ 'train', 'val' ])

    if args.precache:
        precache(dgen_train, 'train dset')
        precache(dgen_test,  'test dset')

    LOGGER.info("Compiling model..")
    np.random.seed(args.seed)

    optimizer = get_optimizer(args.optimizer)
    model     = select_model(args)
    callbacks = get_default_callbacks(args)

    model.compile(
        loss      = args.config.loss,
        optimizer = optimizer,
        metrics   = [ 'mean_relative_error', 'ms_relative_error' ]
    )

    steps_per_epoch = None
    if args.steps_per_epoch is not None:
        steps_per_epoch = min(args.steps_per_epoch, len(dgen_train))

    LOGGER.info("Training model..")
    train_log = model.fit(
        dgen_train,
        epochs          = args.epochs,
        steps_per_epoch = steps_per_epoch,
        validation_data = dgen_test,
        callbacks       = callbacks,
        **get_keras_concurrency_kwargs(args)
    )

    LOGGER.info("Training complete.")

    return return_training_stats(train_log, args.savedir)

