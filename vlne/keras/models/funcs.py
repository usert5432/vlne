"""Functions to construct blocks of layers for `vlne`"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Activation, Add, BatchNormalization, Bidirectional, LayerNormalization,
    Dense, Dropout, Input, LSTM, Masking, TimeDistributed
)

from vlne.consts import DEF_MASK
from vlne.funcs  import unpack_name_args

from .layer_norm import SimpleNorm

def get_normalization_layer(norm, **kwargs):
    norm, basic_kwargs = unpack_name_args(norm)
    kwargs = { **basic_kwargs, **kwargs }

    if norm is None:
        return tf.identity

    if norm == 'batch':
        return BatchNormalization(**kwargs)

    if norm == 'layer':
        return LayerNormalization(**kwargs)

    if norm == 'simple':
        return SimpleNorm(**kwargs)

    raise ValueError("Unknown normalization '%s'" % (norm))

def modify_layer(layer, name, norm = None, dropout = None):
    """Add BatchNorm and/or Dropout on top of `layer`"""

    if dropout is not None:
        name = "%s-dropout" % (name)
        layer = Dropout(dropout, name = name)(layer)

    if norm is not None:
        name  = f'{name}-norm'
        layer = get_normalization_layer(norm, name = name)(layer)

    return layer

def modify_series_layer(
    layer, name, mask = False, norm = None, dropout = None,
    mask_value = DEF_MASK
):
    """Add Mask and/or BatchNorm and/or Dropout on top of series `layer`"""
    if mask:
        name = '%s-masked' % (name)
        layer = Masking(mask_value = mask_value, name = name)(layer)

    if dropout is not None:
        name = "%s-dropout" % (name)
        layer = TimeDistributed(Dropout(dropout), name = name)(layer)

    if norm is not None:
        name  = f'{name}-norm'
        layer = TimeDistributed(
            get_normalization_layer(norm), name = name
        )(layer)

    return layer

def get_inputs(input_groups_scalar, input_groups_vlarr, vlarr_limits):
    inputs_scalar = {}
    inputs_vlarr  = {}

    for (name, columns) in input_groups_scalar.items():
        inputs_scalar[name] = Input(
            shape = (len(columns), ),
            dtype = 'float32',
            name  = name,
        )

    for (name, columns) in input_groups_vlarr.items():
        max_length = None

        if vlarr_limits is not None:
            max_length = vlarr_limits.get(name, None)

        inputs_vlarr[name] = Input(
            shape = (max_length, len(columns)),
            dtype = 'float32',
            name  = name,
        )

    return inputs_scalar, inputs_vlarr

def get_outputs(target_groups, reg, layer):
    """Construct standard `vlne` output layers."""

    outputs = {}

    for (name, columns) in target_groups.items():
        outputs[name] = Dense(
            len(columns),
            name = name,
            kernel_regularizer = reg
        )(layer)

    return outputs

def add_resblock(layer_input, name_prefix, **kwargs):
    """Add a fully connected residual block on top of `layer_input`"""
    input_shape = layer_input.output_shape[1]

    layer_res_fc_1 = Dense(
        input_shape,
        name       = "%s-res-fc1" % (name_prefix),
        activation = 'relu',
        **kwargs
    )(layer_input)

    layer_res_bn_1 = BatchNormalization(
        name = "%s-res-bn1" % (name_prefix)
    )(layer_res_fc_1)

    layer_res_fc_2 = Dense(
        input_shape,
        name       = "%s-res-fc2" % (name_prefix),
        activation = None,
        **kwargs
    )(layer_res_bn_1)

    layer_res_bn_2 = BatchNormalization(
        name = "%s-res-bn2" % (name_prefix),
    )(layer_res_fc_2)

    layer_res_add = Add(
        name = "%s-res-add" % (name_prefix),
    )([layer_res_bn_2, layer_input])

    layer_res_add_act = Activation(
        activation = 'relu',
        name = "%s-res-add-act" % (name_prefix),
    )(layer_res_add)

    return layer_res_add_act

def add_resblocks(layer_input, n, name_prefix, **kwargs):
    """Add `n` fully connected residual blocks on top of `layer_input`"""
    layer = layer_input

    for i in range(n):
        name = "%s-resblock-%d" % (name_prefix, i + 1)
        layer = add_resblock(layer, name, **kwargs)

    return layer

def add_hidden_layers(
    layer_input, layer_sizes, name_prefix, norm, dropout, **kwargs
):
    """Add fully connected layers on top of `layer_input`

    Parameters
    ----------
    layer_input : keras.Layer
        Layer on top of which new layers will be added.
    layer_sizes : list of int
        Shapes of layers to be added
    name_prefix : str
        Prefix to be added to names of new layers.
    norm : str or None
        Name of the normalization layer to use.
    dropout : float or None
        If not None then Dropout layers will be added on top of FC layers
        with a values of dropout of `dropout`.
    kwargs : dict
        Arguments to be passed to the Dense layers constructors.

    Note
    ----
    Do not use `batchnorm` with `dropout` unless you want to be disappointed.

    Returns
    -------
    keras.Layer
        Last layer added on top of `layer_input`
    """
    layer_hidden = layer_input

    for idx,size in enumerate(layer_sizes):
        name = "%s-%d" % (name_prefix, idx + 1)
        layer_hidden = Dense(size, name = name, **kwargs)(layer_hidden)
        layer_hidden = modify_layer(layer_hidden, name, norm, dropout)

    return layer_hidden

def add_hidden_series_layers(
    layer_input, layer_sizes, name_prefix, norm, dropout, **kwargs
):
    """Add fully connected layers on top of series layer `layer_input`
    C.f. `add_hidden_layers` for the description of arguments.

    See Also
    --------
    add_hidden_layers
    """
    layer_hidden = layer_input

    for idx,size in enumerate(layer_sizes):
        name = "%s-%d" % (name_prefix, idx + 1)

        layer_hidden = TimeDistributed(
            Dense(size, **kwargs), name = name
        )(layer_hidden)

        layer_hidden = modify_series_layer(
            layer_hidden, name,
            mask    = False,
            norm    = norm,
            dropout = dropout
        )

    return layer_hidden

def add_stack_of_lstms(
    layer_input, layer_size_dir_pairs, name_prefix, norm, dropout,
    **kwargs
):
    """Add a stack of LSTM layers on top of series layer `layer_input`


    Parameters
    ----------
    layer_input : keras.Layer
        Series layer on top of which new LSTM layers will be added.
    layer_size_dir_pairs : list of (int, str)
        List of pairs that specify number of units and directions of LSTM
        layers to be added.
        Direction can be either 'forward', 'backward' or 'bidirectional'.
    name_prefix : str
        Prefix to be added to names of new LSTM layers.
    norm : str or None
        Name of the normalization layer to use.
    dropout : float or None
        If not None then Dropout layer will be added on top of the last LSTM
        layer with a value of dropout `dropout`.
    kwargs : dict
        Arguments to be passed to the LSTM layers constructors.

    Returns
    -------
    keras.Layer
        Last layer added on top of `layer_input`
    """

    layer_lstm      = layer_input
    is_middle_layer = True

    for idx,size_dir_pair in enumerate(layer_size_dir_pairs):
        name = "%s-%d" % (name_prefix, idx + 1)

        size, direction = size_dir_pair

        if idx == len(layer_size_dir_pairs) - 1:
            is_middle_layer = False

        if direction == 'forward':
            layer_lstm = LSTM(
                size, name = name, return_sequences = is_middle_layer, **kwargs
            )(layer_lstm)

        elif direction == 'backward':
            layer_lstm = LSTM(
                size, name = name, return_sequences = is_middle_layer,
                go_backwards = True, **kwargs
            )(layer_lstm)

        elif direction == 'bidirectional':
            layer_lstm = Bidirectional(
                LSTM(size, return_sequences = is_middle_layer, **kwargs),
                name = name, merge_mode = 'concat',
            )(layer_lstm)
        else:
            raise ValueError("Unknown direction: '%s'" % [direction])

        layer_lstm = modify_series_layer(
            layer_lstm,
            name    = name,
            mask    = False,
            norm    = norm    if is_middle_layer else None,
            dropout = dropout if is_middle_layer else None
        )

    return layer_lstm

