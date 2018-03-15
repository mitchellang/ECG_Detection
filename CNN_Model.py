
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import scipy.io
import gc
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

import keras
from keras.models import Model
from keras.layers import Input, Conv1D, Dense, Flatten, Dropout,MaxPooling1D, Activation, BatchNormalization
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import plot_model
from keras import backend as K
from keras.callbacks import Callback,warnings


def load_data(WINDOW_SIZE):
    matfile = scipy.io.loadmat('trainingset.mat')
    X = matfile['trainset']
    y = matfile['traintarget']

    X =  X[:,0:WINDOW_SIZE]
    return (X,y)

def resnet_model(WINDOW_SIZE):

    OUTPUT_CLASS = 4
    INPUT_SIZE = 1

    conv_filter_size = 64
    #incremental block
    k = 1
    conv_stride = 1
    kernel = 16
    pool_size = 2
    pool_stride = 2
    dropout = 0.5

    input1 = Input(shape=(WINDOW_SIZE,INPUT_SIZE), name='input')

    print(input1)

    l1 = Conv1D(filters = conv_filter_size,
                strides = conv_stride,
                padding = 'same',
                kernel_size = kernel,
                kernel_initializer = 'he_normal')(input1)
    l1 = BatchNormalization()(l1)
    l1 = Activation('relu')(l1)

    l2 = Conv1D(filters = conv_filter_size,
                strides = conv_stride,
                padding = 'same',
                kernel_size = kernel,
                kernel_initializer = 'he_normal')(l1)
    l2 = BatchNormalization()(l2)
    l2 = Activation('relu')(l2)
    l2 = Dropout(dropout)(l2)
    l2 = Conv1D(filters = conv_filter_size,
                strides = conv_stride,
                padding = 'same',
                kernel_size = kernel,
                kernel_initializer = 'he_normal')(l2)
    l2 = MaxPooling1D(pool_size = pool_size,
                      strides = pool_stride)(l2)

    l3 = MaxPooling1D(pool_size = pool_size,
                      strides = pool_stride)(l1)

    l1 = keras.layers.add([l2, l3])
    del l2, l3

    p = True

    for i in range(15):

        if (i % 4 == 0) and (i > 0):  # increment k on every fourth residual block
            k += 1
            # increase depth by 1x1 Convolution case dimension shall change
            lshort = Conv1D(filters = conv_filter_size * k, kernel_size = 1)(l1)
        else:
            lshort = l1
            # Left branch (convolutions)
        # notice the ordering of the operations has changed
        l2 = BatchNormalization()(l1)
        l2 = Activation('relu')(l2)
        l2 = Dropout(dropout)(l2)
        l2 = Conv1D(filters = conv_filter_size * k,
                    kernel_size = kernel,
                    padding = 'same',
                    strides = conv_stride,
                    kernel_initializer = 'he_normal')(l2)
        l2 = BatchNormalization()(l2)
        l2 = Activation('relu')(l2)
        l2 = Dropout(dropout)(l2)
        l2 = Conv1D(filters = conv_filter_size * k,
                    kernel_size = kernel,
                    padding = 'same',
                    strides = conv_stride,
                    kernel_initializer = 'he_normal')(l2)
        if p:
            l2 = MaxPooling1D(pool_size = pool_size, strides = pool_stride)(l2)

            # Right branch: shortcut connection
        if p:
            l3 = MaxPooling1D(pool_size = pool_size, strides = pool_stride)(lshort)
        else:
            l3 = lshort  # pool or identity
        # Merging branches
        l1 = keras.layers.add([l2, l3])
        # change parameters
        p = not p  # toggle pooling




    l1 = BatchNormalization()(l1)
    l1 = Activation('relu')(l1)
    l1 = Flatten()(l1)

    output = Dense(OUTPUT_CLASS, activation='softmax')(l1)

    model = Model(inputs = input1, outputs = output)
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.9, amsgrad=False)
    model.compile(optimizer = adam,
                  loss = 'categorical_crossentropy',
                  metrics = ['accuracy'])

    # plot_model(model, to_file = 'model.png')
    return model

def evaluate_model():
    FS = 300
    WINDOW_SIZE = 30 * FS

    #Testing epochs = 10
    #actual epochs = 20
    epochs = 10
    batch_size = 64
    (X,y) = load_data(WINDOW_SIZE)

    # For Testing
    # X = X[:200]
    # y = y[:200]

    X = np.expand_dims(X, axis = 2)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

    model = resnet_model(WINDOW_SIZE)

    print(x_train.shape)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, verbose=1)
        ]

    model.fit(x_train,y_train,
              validation_data = (x_test,y_test),
              epochs = epochs,
              batch_size = batch_size,
              callbacks = callbacks
              )

    model.save_weights('model_test.hdf5')

    print("prediction")

    model.load_weights('model_test.hdf5')

    ypred = model.predict(x_test)
    ypred = np.argmax(ypred, axis = 1)
    ytrue = np.argmax(y_test, axis = 1)

    print(x_test.shape)
    print(np.savetxt('ypred.csv', ypred))
    print(np.savetxt('ytrue.csv',ytrue))

    results = precision_recall_fscore_support(ytrue, ypred)

    print(results)

    return results

sess = tf.Session()
seed = 7
np.random.seed(seed)

evaluate_model()