from keras import backend as K
from keras.engine.base_layer import Layer, InputSpec

class SimpleNorm(Layer):
    """Layer that standartizes data by using moving mean/variance
        
    Parameters
    ----------
    momentum : float
        Momentum for the moving mean and the moving variance.
        Default: 0.99
    epsilon : float
        Number that is added to the denominator to prevent division by zero.
    """

    def __init__(self, momentum = 0.99, epsilon = 1e-3, **kwargs):
        super().__init__(**kwargs)

        self._momentum = momentum
        self._epsilon  = epsilon
        self.supports_masking = True

    def build(self, input_shape):
        if len(input_shape) != 2:
            raise ValueError(
                'SimpleNorm inputs should have rank 2. Got %s' % (
                    input_shape,
                )
            )

        n_features = input_shape[-1]
        shape      = (n_features, )

        self.input_spec = InputSpec(ndim = 2, axes = { 1 : n_features })

        self.moving_mean = self.add_weight(
            shape       = shape,
            name        = 'moving_mean',
            initializer = 'zeros',
            trainable   = False
        )

        self.moving_var = self.add_weight(
            shape       = shape,
            name        = 'moving_var',
            initializer = 'ones',
            trainable   = False
        )

        self.built = True

    def update_moving_averages(self, inputs):
        # inputs : (N, n_features)

        # mean : (n_features,)
        # var  : (n_features,)
        mean = K.mean(inputs, axis = 0)
        var  = K.var(inputs,  axis = 0)

        self.add_update(
            [
                K.moving_average_update(self.moving_mean, mean, self.momentum),
                K.moving_average_update(self.moving_var,  std,  self.momentum),
            ],
            inputs
        )

    def call(self, inputs, training = None):
        # inputs : (N, n_features)
        if training:
            self.update_moving_averages(inputs)

        return K.batch_normalization(
            inputs,
            self.moving_mean,
            self.moving_var,
            beta    = 0,
            gamma   = 1,
            axis    = 1,
            epsilon = self._epsilon
        )

