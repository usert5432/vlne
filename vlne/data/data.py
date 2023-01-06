"""
A collection of routines to simplify data handling.
"""

import itertools
import logging
import os

from vlndata.data_frame import (
    select_frame, ShuffleFrame, VarFrame, train_test_split
)
from vlndata.dataset    import construct_dataset_from_data_frame, SPLIT_INDEX

from vlne.data.data_generator import DataGenerator
from vlne.data.data_generator.funcs.weights import flat_weights

from vlne.funcs import unpack_name_args

LOGGER  = logging.getLogger('vlne.data')

def select_vlne_frame(name, path, datadir, **args):
    path = os.path.join(datadir, path)
    return select_frame({ 'name' : name, 'path' : path, **args})

def parse_extra_vars(extra_vars):
    result = {}

    for (column, var_spec) in extra_vars.items():
        name, kwargs = unpack_name_args(var_spec)
        if name == 'flat':
            result[column] = \
                lambda df, kwargs = kwargs : flat_weights(df, **kwargs)
        else:
            raise ValueError(f"Unknown variable {name}")

    return result

def create_data_frame(data_config, datadir = None):
    df = select_vlne_frame(**data_config.frame, datadir = datadir)

    if data_config.extra_vars is not None:
        extra_vars = parse_extra_vars(data_config.extra_vars)
        df = VarFrame(df, variables = extra_vars, lazy = False)

    if data_config.shuffle:
        df = ShuffleFrame(df, seed = data_config.seed)

    if (data_config.test_size is None) and (data_config.val_size is None):
        return df

    return train_test_split(df, data_config.val_size, data_config.test_size)

def create_datasets_from_single_df(df, data_config, cache, splits):
    scalar_groups = {
        **data_config.input_groups_scalar, **data_config.target_groups
    }

    if isinstance(splits, (tuple, list)):
        assert len(splits) == 1
        split = splits[0]
    else:
        split = splits

    return [ construct_dataset_from_data_frame(
        df, cache, split, scalar_groups,
        vlarr_groups    = data_config.vlarr_groups,
        vlarr_limits    = data_config.vlarr_limits,
        transform_train = data_config.transform_train,
        transform_test  = data_config.transform_test
    ), ]

def create_datasets_from_df_list(df_list, data_config, cache, splits):
    scalar_groups = {
        **data_config.input_groups_scalar, **data_config.target_groups
    }

    if not isinstance(splits, (tuple, list)):
        splits = [ splits, ]

    result = []

    for split in splits:
        result.append(
            construct_dataset_from_data_frame(
                df_list[SPLIT_INDEX[split]], cache, split, scalar_groups,
                vlarr_groups    = data_config.input_groups_vlarr,
                vlarr_limits    = data_config.vlarr_limits,
                transform_train = data_config.transform_train,
                transform_test  = data_config.transform_test
            )
        )

    return result

def create_datasets(df, data_config, cache, splits):
    if not isinstance(df, (tuple, list)):
        return create_datasets_from_single_df(df, data_config, cache, splits)

    return create_datasets_from_df_list(df, data_config, cache, splits)

def create_data_generators_from_datasets(dset_list, data_config, batch_size):
    LOGGER.info("Creating data generators with batch size %d", batch_size)
    input_groups = list(itertools.chain(
        data_config.input_groups_scalar.keys(),
        data_config.input_groups_vlarr.keys()
    ))

    target_groups = list(data_config.target_groups.keys())

    return [
        DataGenerator(
            x, input_groups, target_groups, batch_size, data_config.weights
        )
        for x in dset_list
    ]

def create_data_generators(data_config, batch_size, splits, datadir, cache):
    df_list   = create_data_frame(data_config, datadir)
    dset_list = create_datasets(df_list, data_config, cache, splits)

    dgen_list = create_data_generators_from_datasets(
        dset_list, data_config, batch_size
    )

    # pylint: disable = import-outside-toplevel
    from vlne.data.data_generator.keras_sequence import KerasSequence
    dgen_list = [ KerasSequence(x) for x in dgen_list ]

    return dgen_list

def load_data(args, splits):
    return create_data_generators(
        data_config = args.config.data,
        batch_size  = args.config.batch_size,
        splits      = splits,
        datadir     = args.root_datadir,
        cache       = args.cache,
    )

