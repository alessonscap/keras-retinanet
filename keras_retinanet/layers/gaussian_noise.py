from keras import backend as K
from keras.engine.topology import Layer
import keras

class GaussianNoiseOur(Layer):
    """Apply additive zero-centered Gaussian noise.
    This is useful to mitigate overfitting
    (you could see it as a form of random data augmentation).
    Gaussian Noise Our (GS) is a natural choice as corruption process
    for real valued inputs.
    As it is a regularization layer, it is only active at training time.
    # Arguments
        stddev: float, standard deviation of the noise distribution.
        scale:  float, scaling multiplier
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as input.
    """

    def __init__(self, stddev,scale=1, **kwargs):
        super(GaussianNoiseOur, self).__init__(**kwargs)
        self.supports_masking = True
        self.stddev = stddev
        self.scale = scale

    def call(self, inputs, training=None):
        def noised():
            return inputs + self.scale*K.random_normal(shape=K.shape(inputs),
                                            mean=0.,
                                            stddev=self.stddev)
        return K.in_train_phase(noised, inputs, training=training)


    def get_config(self):
        config = {'stddev': self.stddev}
        base_config = super(GaussianNoiseOur, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape