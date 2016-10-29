__author__ = "Nadimozzaman Pappo"
__github__ = "http://github.com/mnpappo"

"""
This script use keras & minist dataset to test random Dropout effects on result.
"""
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.utils.visualize_util import plot
# from matplotlib.cm import binary
import matplotlib.pyplot as plt
import random
from theano.printing import Print
import theano
from theano import tensor as T
from theano import tensor
#from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.shared_randomstreams import RandomStreams
from theano.tensor.raw_random import RandomStreamsBase, RandomFunction, _infer_ndim_bcast
from theano.compile.sharedvalue import (SharedVariable, shared_constructor,
                                        shared)


def new_binomial(random_state, size=None, n=1, p=None, ndim=None,
             dtype='int64', prob=None):
    """
    Sample n times with probability of success prob for each trial,
    return the number of successes.
    If the size argument is ambiguous on the number of dimensions, ndim
    may be a plain integer to supplement the missing information.
    If size is None, the output shape will be determined by the shapes
    of n and prob.
    """



    n = theano.compile.sharedvalue.shared(n)
    p = theano.compile.sharedvalue.shared(p)

    ndim, size, bcast = _infer_ndim_bcast(ndim, size, n, p)


    if n.dtype == 'int64':
        try:
            np.random.binomial(n=np.asarray([2, 3, 4], dtype='int64'), p=np.asarray([.1, .2, .3], dtype='float64'))
            print("Using xxxxxxx")
        except TypeError:
            # THIS WORKS AROUND A NUMPY BUG on 32bit machine
            n = tensor.cast(n, 'int32')
    op = RandomFunction('binomial', tensor.TensorType(dtype=dtype, broadcastable=(False,) * ndim))
    return op(random_state, size, n, p)


class NewRandomStreamsBase(RandomStreamsBase):

    def new_binomial(self, size=None, n=1, p=None, ndim=None, dtype='int64',
                 prob=None):
        """
        Sample n times with probability of success p for each trial and
        return the number of successes.
        If the size argument is ambiguous on the number of dimensions,
        ndim may be a plain integer to supplement the missing information.
        """
        print("Using new Rand")

        return self.gen(new_binomial, size, n, p, ndim=ndim, dtype=dtype)


class NewRandomStreams(NewRandomStreamsBase):
    """
    Module component with similar interface to numpy.random
    (numpy.random.RandomState)
    Parameters
    ----------
    seed: None or int
        A default seed to initialize the RandomState
        instances after build.  See `RandomStreamsInstance.__init__`
        for more details.
    """

    def updates(self):
        return list(self.state_updates)

    def __init__(self, seed=None):
        super(NewRandomStreams, self).__init__()
        # A list of pairs of the form (input_r, output_r).  This will be
        # over-ridden by the module instance to contain stream generators.
        self.state_updates = []
        # Instance variable should take None or integer value. Used to seed the
        # random number generator that provides seeds for member streams.
        self.default_instance_seed = seed
        # numpy.RandomState instance that gen() uses to seed new streams.
        self.gen_seedgen = np.random.RandomState(seed)

    def seed(self, seed=None):
        """
        Re-initialize each random stream.
        Parameters
        ----------
        seed : None or integer in range 0 to 2**30
            Each random stream will be assigned a unique state that depends
            deterministically on this value.
        Returns
        -------
        None
        """
        if seed is None:
            seed = self.default_instance_seed

        seedgen = np.random.RandomState(seed)
        for old_r, new_r in self.state_updates:
            old_r_seed = seedgen.randint(2 ** 30)
            old_r.set_value(np.random.RandomState(int(old_r_seed)),
                            borrow=True)

    def __getitem__(self, item):
        """
        Retrieve the numpy RandomState instance associated with a particular
        stream.
        Parameters
        ----------
        item
            A variable of type RandomStateType, associated
            with this RandomStream.
        Returns
        -------
        numpy RandomState (or None, before initialize)
        Notes
        -----
        This is kept for compatibility with `tensor.randomstreams.RandomStreams`.
        The simpler syntax ``item.rng.get_value()`` is also valid.
        """
        return item.get_value(borrow=True)

    def __setitem__(self, item, val):
        """
        Set the numpy RandomState instance associated with a particular stream.
        Parameters
        ----------
        item
            A variable of type RandomStateType, associated with this
            RandomStream.
        val : numpy RandomState
            The new value.
        Returns
        -------
        None
        Notes
        -----
        This is kept for compatibility with `tensor.randomstreams.RandomStreams`.
        The simpler syntax ``item.rng.set_value(val)`` is also valid.
        """
        item.set_value(val, borrow=True)

    def gen(self, op, *args, **kwargs):
        """
        Create a new random stream in this container.
        Parameters
        ----------
        op
            A RandomFunction instance to
        args
            Interpreted by `op`.
        kwargs
            Interpreted by `op`.
        Returns
        -------
        Tensor Variable
            The symbolic random draw part of op()'s return value.
            This function stores the updated RandomStateType Variable
            for use at `build` time.
        """
        seed = int(self.gen_seedgen.randint(2 ** 30))
        random_state_variable = shared(np.random.RandomState(seed))
        # Add a reference to distinguish from other shared variables
        random_state_variable.tag.is_rng = True
        new_r, out = op(random_state_variable, *args, **kwargs)
        out.rng = random_state_variable
        out.update = (random_state_variable, new_r)
        self.state_updates.append(out.update)
        random_state_variable.default_update = new_r
        return out


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
    # if seed is None:
    #     seed = np.random.randint(1, 10e6)

    rng = NewRandomStreams(seed=seed)
    retain_prob = level


    # print("--------")
    # print("Shape is:", x.shape)
    # print("Shape is:", x.shape)
    # print("--------")
    # retain_prob = np.random.uniform(0,1,128)
    # retain_prob = retain_prob.astype(x.dtype)

    if noise_shape is None:
        random_tensor = rng.new_binomial(x.shape, p=retain_prob, dtype=x.dtype)
    else:
        random_tensor = rng.new_binomial(noise_shape, p=retain_prob, dtype=x.dtype)
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


batch_size = 128
nb_classes = 10
nb_epoch = 2

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

np.random.seed(1337)  # for reproducibility

# taking Dropout
# dropout = input("Please give a Dropout value(0-1): ")
# while dropout < 0 or dropout > 1:
#     print("Dropout value must be in range 0-1.")
#     dropout = input("Please give a Dropout value(0-1): ")
#
# print("Using Dropout {0}".format(dropout))


def get_random_dropout(a, b):
    x = np.random.uniform(0,1,(128))
    x = x.astype('float32')
    # return round(random.uniform(a, b), 2)
    return x

# plotting/visualizing layers weight
def plot_layer(layer, x, y):
    layer_config = layer.get_config()
    print("Layer Name: ", layer_config['name'])
    layer_weight, layer_bias = layer.get_weights()
    figure = plt.figure()
    for i in range(len(layer_weight)):
        ax = figure.add_subplot(y, x, i+1)
        ax.matshow(layer_weight[i][0], cmap=binary)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    figure.set_tight_layout(True)
    plt.show()
    # figure.savefig('test.png')

def load_dataset():
    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # get input data shape
    if K.image_dim_ordering() == 'th':
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    # print('X_train shape:', X_train.shape)
    # print(X_train.shape[0], 'train samples')
    # print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    return X_train, Y_train, X_test, Y_test, input_shape


def make_network(input_shape):
    model = Sequential()
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='valid',
                            input_shape=input_shape))

    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))

    dropout = get_random_dropout(0,.4)
    layer_info = "MaxPooling2D"
    print("using dropout: " + str(dropout) + " after layer: " + layer_info)
    model.add(RandomDropout(dropout))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))

    dropout = get_random_dropout(0,.6)
    layer_info = "Activation"
    print("using dropout: " + str(dropout) + " after layer: " + layer_info)
    model.add(RandomDropout(dropout))

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    return model

# training model with SGD with momentum
def train_model(model, X_train, Y_train, X_test, Y_test):
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              verbose=1, validation_data=(X_test, Y_test))
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

# saving the trained model & mnist model architecture
def save_model(model):
   model_json = model.to_json()
   open('data/mnist_architecture.json', 'w').write(model_json)
   model.save_weights('data/mnist_weights.h5', overwrite=True)


if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test, input_shape = load_dataset()
    model = make_network(input_shape)
    trained_model = train_model(model, X_train, Y_train, X_test, Y_test)
    print("Training completed. Saving the model.")
    save_model(model)
    plot(model, to_file='data/model.png')
    # visualizing weights for first layer
    # plot_layer(model.layers[0], 8, 4)
