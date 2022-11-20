"""
A collection of functions to setup keras training.
"""

import tensorflow as tf
from tensorflow import keras

from vlne.funcs import unpack_name_args
from vlne.keras.callbacks import TrainTime
from vlne.keras.models    import (
    flattened_model, model_lstm_v1, model_lstm_v2, model_lstm_v3,
    model_lstm_v4, model_slice_linear, model_lstm_v3_stack,
    model_trans_v1
)

def get_optimizer(optimizer):
    name, kwargs = unpack_name_args(optimizer)

    if name.lower() == 'rmsprop':
        return keras.optimizers.RMSprop(**kwargs)

    elif name.lower() == 'adam':
        return keras.optimizers.Adam(**kwargs)

    else:
        raise ValueError("Unknown optimizer: %s" % (optimizer))

def get_schedule(schedule):
    name, kwargs = unpack_name_args(schedule)
    kwargs['verbose'] = True

    if name.lower() == 'standard':
        return keras.callbacks.ReduceLROnPlateau(**kwargs)

    elif name.lower() == 'custom':
        return keras.callbacks.LearningRateScheduler(**kwargs)

    else:
        raise ValueError("Unknown schedule: %s" % (schedule))

def get_early_stop(early_stop):
    name, kwargs = unpack_name_args(early_stop)
    kwargs['verbose'] = True

    if name.lower() == 'standard':
        return keras.callbacks.EarlyStopping(**kwargs)

    else:
        raise ValueError("Unknown early stoping: %s" % (early_stop))

def get_default_callbacks(args):
    cb_checkpoint = keras.callbacks.ModelCheckpoint(
        "%s/model.h5" % args.savedir,
        monitor           = 'val_loss',
        verbose           = 0,
        save_best_only    = args.save_best,
    )

    cb_logger     = keras.callbacks.CSVLogger("%s/log.csv" % args.savedir)
    cb_time       = TrainTime()
    cb_schedule   = get_schedule(args.schedule)
    cb_early_stop = get_early_stop(args.early_stop)

    callbacks = [ cb_time, cb_checkpoint, cb_logger ]

    if cb_schedule is not None:
        callbacks.append(cb_schedule)

    if cb_early_stop is not None:
        callbacks.append(cb_early_stop)

    return callbacks

def get_regularizer(regularizer):
    name, kwargs = unpack_name_args(regularizer)

    if name.lower() == 'l1':
        return keras.regularizers.l1(**kwargs)

    if name.lower() == 'l2':
        return keras.regularizers.l2(**kwargs)

    if name.lower() == 'l1_l2':
        return keras.regularizers.l1_l2(**kwargs)

    raise ValueError("Unknown regularizer: %s" % (regularizer))

def select_model(args):
    # pylint: disable=too-many-return-statements
    name, kwargs = unpack_name_args(args.model)
    kwargs = {
        'reg'                 : get_regularizer(args.regularizer),
        'input_groups_scalar' : args.data.input_groups_scalar,
        'input_groups_vlarr'  : args.data.input_groups_vlarr,
        'target_groups'       : args.data.target_groups,
        'vlarr_limits'        : args.data.vlarr_limits,
        **kwargs
    }

    if name == 'lstm_v1':
        return model_lstm_v1(**kwargs)
    if name == 'lstm_v2':
        return model_lstm_v2(**kwargs)
    if name == 'lstm_v3':
        return model_lstm_v3(**kwargs)
    if name == 'lstm_v4':
        return model_lstm_v4(**kwargs)
    if name == 'lstm_v3_stack':
        return model_lstm_v3_stack(**kwargs)
    if name == 'slice_linear':
        return model_slice_linear(**kwargs)
    if name == 'flattened':
        return flattened_model(**kwargs)
    if name == 'trans_v1':
        return model_trans_v1(**kwargs)
    else:
        raise ValueError("Unknown model name: %s" % (args.model))

def get_keras_concurrency_kwargs(args):
    result = {
        'workers' : 0,
        'use_multiprocessing' : True,
    }

    if (args.workers is None) or (args.workers < 1):
        return result

    result['workers'] = args.workers

    return result

def limit_tf_memory_growth():
    gpus = tf.config.list_physical_devices('GPU')

    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

