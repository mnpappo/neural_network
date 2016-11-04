import numpy as np
from keras.layers import Dropout
from keras import backend as K
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams


def get_random_dropout(a, b, shape):
    x = np.random.uniform(0, 1, shape)
    x = x.astype('float32')
    return x

def randdrop(x, level, noise_shape=None, seed=None):
    '''Sets entries in `x` to zero at random,
    while scaling the entire tensor.
    # Arguments
        x: tensor
        level: fraction of the entries in the tensor
            that will be set to 0.
        noise_shape: shape for randomly generated keep/drop flags,
            must be broadcastable to the shape of `x`
        seed: random seed to ensure determinism.
    '''
    # if level < 0. or level >= 1:
    #     raise Exception('Dropout level must be in interval [0, 1[.')
    if seed is None:
        seed = np.random.randint(1337)

    rng = RandomStreams(seed=seed)
    retain_prob = 1 - level

    if noise_shape is None:
        random_tensor = rng.binomial(x.shape, p=retain_prob, dtype=x.dtype)
    else:
        random_tensor = rng.binomial(noise_shape, p=retain_prob, dtype=x.dtype)
        random_tensor = T.patternbroadcast(random_tensor, [dim == 1 for dim in noise_shape])

    x *= random_tensor
    x /= retain_prob
    return x

class RandomDropout(Dropout):
    print("Dropout calling from local")
    def __init__(self, p, **kwargs):
        self.p = p
        # if 0. < self.p.all() < 1.:
        self.uses_learning_phase = True
        self.supports_masking = True
        super(Dropout, self).__init__(**kwargs)

    def call(self, x, mask=None):
        # if 0. < self.p < 1.:
        x = K.in_train_phase(randdrop(x, level=self.p), x)
        return x
