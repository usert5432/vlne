"""
Functions to construct Transformer based models.
"""

import tensorflow as tf

# pylint: disable=no-name-in-module
from tensorflow.keras.layers import Dense, MultiHeadAttention, TimeDistributed

from vlne.consts import DEF_MASK
from .funcs import (
    modify_series_layer, add_hidden_series_layers, get_normalization_layer
)
# get_inputs, get_outputs, add_hidden_layers,

@tf.keras.utils.register_keras_serializable('vlne.keras.models')
class PositionWiseFFN(tf.keras.layers.Layer):

    def __init__(self, num_hidden, ffn_num_hidden, activation, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

        self.num_hidden     = num_hidden
        self.ffn_num_hidden = ffn_num_hidden
        self.activation     = activation

        self.fc1 = TimeDistributed(
            Dense(ffn_num_hidden, activation = activation)
        )
        self.fc2 = TimeDistributed(Dense(num_hidden))

    def get_config(self):
        config = super().get_config()

        config.update({
            "num_hidden"     : self.num_hidden,
            "ffn_num_hidden" : self.ffn_num_hidden,
            "activation"     : self.activation,
        })

        return config


    def call(self, inputs, **kwargs):
        y = self.fc1(inputs, **kwargs)

        return self.fc2(y, **kwargs)

@tf.keras.utils.register_keras_serializable('vlne.keras.models')
class TransformerEncoderBlock(tf.keras.layers.Layer):
    # pylint: disable=too-many-instance-attributes

    def __init__(
        self, num_hidden, ffn_num_hidden, num_heads, activation,
        rezero = True, norm = 'layer', **kwargs
    ):
        super().__init__(**kwargs)
        self.supports_masking = True

        with tf.name_scope(self.name):
            self.norm1 = TimeDistributed(get_normalization_layer(norm))
            self.atten = MultiHeadAttention(
                num_heads = num_heads, key_dim = num_hidden
            )

            self.norm2 = TimeDistributed(get_normalization_layer(norm))
            self.pffn = PositionWiseFFN(num_hidden, ffn_num_hidden, activation)

            self.num_hidden     = num_hidden
            self.ffn_num_hidden = ffn_num_hidden
            self.num_heads      = num_heads
            self.activation     = activation
            self.rezero         = rezero
            self.norm           = norm

            if rezero:
                self.re_alpha = tf.Variable(
                    0., dtype = tf.float32, trainable = True
                )
            else:
                self.re_alpha = tf.constant(1., dtype = tf.float32)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_hidden'     : self.num_hidden,
            'ffn_num_hidden' : self.ffn_num_hidden,
            'num_heads'      : self.num_heads,
            'activation'     : self.activation,
            'rezero'         : self.rezero,
            'norm'           : self.norm,
        })

        return config

    def call(self, x, mask = None, **kwargs):
        y1 = self.norm1(x, mask = mask, **kwargs)
        y1 = self.atten(y1, y1, y1, **kwargs)
        y1 = x + self.re_alpha * y1

        y2 = self.norm2(y1, mask = mask, **kwargs)
        y2 = self.pffn(y2, mask = mask, **kwargs)
        y2 = y1 + self.re_alpha * y2

        return y2

@tf.keras.utils.register_keras_serializable('vlne.keras.models')
class GlobalTransPooling(tf.keras.layers.Layer):

    def __init__(self, pool_type = 'sum', axis = 1, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self._axis = axis
        self._type = pool_type

        if pool_type not in [ 'sum', 'max', 'first' ]:
            raise ValueError("Unknown pooling type: '%s'" % pool_type)

    def call(self, inputs, mask = None, **_kwargs):
        if mask is None:
            mask = 1
        else:
            mask = tf.broadcast_to(
                tf.expand_dims(tf.cast(mask, "float32"), -1),
                tf.shape(inputs)
            )

        if self._type == 'sum':
            return tf.reduce_sum(mask * inputs, axis = self._axis)

        if self._type == 'max':
            return tf.reduce_max(mask * inputs, axis = self._axis)

        if self._type == 'first':
            return inputs[:, 0, :]

        return inputs

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'axis'      : self._axis,
            'pool_type' : self._type,
        })

        return config

    def compute_mask(self, inputs, mask):
        # pylint: disable=unused-argument
        # pylint: disable=no-self-use
        return None

def prepare_atten_vect_inputs(
    branch_label, input_layer, num_hidden, reg, mask_value = DEF_MASK
):
    input_shape = [ num_hidden, ]

    input_layer = modify_series_layer(
        input_layer, 'input_%s' % (branch_label,),
        mask = True, norm = 'batch', mask_value = mask_value
    )

    layer_hidden_pre = add_hidden_series_layers(
        input_layer, input_shape, "hidden_pre_%s" % (branch_label,),
        norm = None, dropout = None, activation = 'relu',
        kernel_regularizer = reg,
    )

    return layer_hidden_pre

def aggregate_trans_output(agg_type, inputs):
    if agg_type == 'avg':
        return tf.keras.layers.GlobalAveragePooling1D()(inputs)

    return GlobalTransPooling(agg_type)(inputs)

def model_trans_v1(
    num_heads          = 8,
    num_hidden         = 128,
    ffn_num_hidden     = 128,
    num_blocks         = 3,
    layers_post        = [],
    max_prongs         = None,
    reg                = None,
    agg_type           = 'max',
    activation         = 'gelu',
    rezero             = True,
    norm               = 'layer',
    vars_input_slice   = None,
    vars_input_png3d   = None,
    vars_input_png2d   = None,
    var_target_total   = None,
    var_target_primary = None,
):
    # pylint: disable=dangerous-default-value
    # pylint: disable=unused-argument
    # pylint: disable=pointless-string-statement
    # pylint: disable=unreachable

    # TODO: port to new data config

    raise NotImplementedError
    """
    assert(vars_input_png2d is None)

    inputs = get_inputs(
        vars_input_slice, vars_input_png3d, vars_input_png2d, max_prongs
    )
    # pylint: disable=unbalanced-tuple-unpacking
    input_slice, input_png3d = inputs

    input_png3d = prepare_atten_vect_inputs(
        'png3d', input_png3d, num_hidden, reg
    )
    input_slice = prepare_atten_vect_inputs(
        'slice', tf.keras.layers.RepeatVector(1)(input_slice), num_hidden, reg
    )

    layer_merged = tf.keras.layers.Concatenate(axis = 1)(
        [ input_slice, input_png3d ]
    )

    layer_trans = layer_merged
    for i in range(num_blocks):
        layer_trans = TransformerEncoderBlock(
            num_hidden, ffn_num_hidden, num_heads, activation, rezero, norm,
            name = 'trans_encoder_%d' % i
        )(layer_trans)

    layer_agg  = aggregate_trans_output(agg_type, layer_trans)
    layer_post = add_hidden_layers(
        layer_agg, layers_post, "hidden_post", norm = None, dropout = None,
        activation = activation, kernel_regularizer = reg,
    )

    outputs = get_outputs(
        var_target_total, var_target_primary, reg, layer_post
    )

    return tf.keras.Model(inputs = inputs, outputs = outputs)
    """


