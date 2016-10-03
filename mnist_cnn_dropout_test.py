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
from matplotlib.cm import binary
import matplotlib.pyplot as plt
import random

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
    return round(random.uniform(a, b), 2)

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
    model.add(Dropout(dropout))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))

    dropout = get_random_dropout(0,.6)
    layer_info = "Activation"
    print("using dropout: " + str(dropout) + " after layer: " + layer_info)
    model.add(Dropout(dropout))

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
