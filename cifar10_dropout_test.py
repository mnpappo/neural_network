from __future__ import print_function
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD

batch_size = 32
nb_classes = 10
nb_epoch = 1
img_channels = 3
img_rows, img_cols = 32, 32

def load_dataset():
   # the data, shuffled and split between train and test sets
   (X_train, y_train), (X_test, y_test) = cifar10.load_data()
   print('X_train shape:', X_train.shape)
   print(X_train.shape[0], 'train samples')
   print(X_test.shape[0], 'test samples')

   # convert class vectors to binary class matrices
   Y_train = np_utils.to_categorical(y_train, nb_classes)
   Y_test = np_utils.to_categorical(y_test, nb_classes)

   X_train = X_train.astype('float32')
   X_test = X_test.astype('float32')
   X_train /= 255
   X_test /= 255

   return X_train, Y_train, X_test, Y_test

def make_network():
   model = Sequential()

   model.add(Convolution2D(32, 3, 3, border_mode='same',
                           input_shape=(img_channels, img_rows, img_cols)))
   model.add(Activation('relu'))
   model.add(Convolution2D(32, 3, 3))
   model.add(Activation('relu'))
   model.add(MaxPooling2D(pool_size=(2, 2)))
   model.add(Dropout(0.25))

   model.add(Convolution2D(64, 3, 3, border_mode='same'))
   model.add(Activation('relu'))
   model.add(Convolution2D(64, 3, 3))
   model.add(Activation('relu'))
   model.add(MaxPooling2D(pool_size=(2, 2)))
   model.add(Dropout(0.25))

   model.add(Flatten())
   model.add(Dense(512))
   model.add(Activation('relu'))
   model.add(Dropout(0.5))
   model.add(Dense(nb_classes))
   model.add(Activation('softmax'))

   return model

def train_model(model, X_train, Y_train, X_test, Y_test):

   sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
   # model.compile(loss='categorical_crossentropy', optimizer=sgd)
   model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

   # model.fit(X_train, Y_train, nb_epoch=nb_epoch, batch_size=batch_size, validation_split=0.1, show_accuracy=True, verbose=1)
   model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(X_test, Y_test), shuffle=True)

   print('Testing...')
   res = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=1)
   print("--------")
   print(res)
   print("--------")
   print('Test accuracy: {0}'.format(res[1]))


def save_model(model):

   model_json = model.to_json()
   open('data/cifar10_architecture.json', 'w').write(model_json)
   model.save_weights('data/cifar10_weights.h5', overwrite=True)

if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = load_dataset()
    model = make_network()
    trained_model = train_model(model, X_train, Y_train, X_test, Y_test)
    save_model(model)
