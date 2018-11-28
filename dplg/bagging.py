# coding: utf-8

from __future__ import division, print_function, unicode_literals, absolute_import
import tensorflow as tf 
import numpy as np 
import math
from dpdl import input
from keras.layers import Input, Dense 
from keras.models import Model
from sklearn.preprocessing import OneHotEncoder


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('dataset', 'mnist', 'dataset name, mnist, svhn, or cifar10')
tf.flags.DEFINE_string('data_dir', './data_dir', 'file dir path to store data')
tf.flags.DEFINE_integer('max_steps', 3000, 'max steps train students')
tf.flags.DEFINE_integer('batch_size', 1800, 'batch_size')
tf.flags.DEFINE_float('epsilon', 0.15, 'privacy epsilon')
tf.flags.DEFINE_integer('nb_labels', 10, 'number of dataset labels')
tf.flags.DEFINE_float('learning_rate', 0.001, 'learning rate of mlp')
tf.flags.DEFINE_integer('dim', 32, 'dim of encoding image')
tf.flags.DEFINE_integer('epochs', 0, 'epochs of training autoencoder')



def load_data(dataset):
    if dataset == 'svhn':
        train_data,train_labels,test_data,test_labels = input.ld_svhn(extended=True)
    elif dataset == 'cifar10':
        train_data, train_labels, test_data, test_labels = input.ld_cifar10()
    elif dataset == 'mnist':
        train_data, train_labels, test_data, test_labels = input.ld_mnist()
    else:
        print("Check value of dataset flag")
        return False
    return train_data, train_labels, test_data, test_labels

def stackautoencoder(dim, encoding_dim):
    input_img = Input(shape=(dim,))
    encoded = Dense(128, activation='relu')(input_img)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(32, activation='relu')(encoded)
    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(dim, activation='sigmoid')(decoded)

    autoencoder = Model(input_img, decoded)    
    encoder = Model(input_img, encoded)
    # encoded_input = Input(shape=(32, ))
    # decoder_layer = autoencoder.layers[-1]
    # decoder = Model(encoded_input, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    return encoder, None, autoencoder

def train_ae(x_train, x_test, dim1):
    data_len, dim = x_train.shape
    # dim1 = 32
    # dim2 = 32
    # encoder1, decoder1, ae1 = autoencoder(dim, dim1)
    encoder, _, ae = stackautoencoder(dim, dim1)
    ae.fit(x_train, x_train, epochs=FLAGS.epochs, batch_size=256, shuffle=True, validation_data=(x_test, x_test))
    e_train = encoder.predict(x_train)
    e_test = encoder.predict(x_test)
    # plot(e_train, decoder, x_train)
    return e_train, e_test

def get_noise(data_len, dim1):
    # xy = x_train * y_train
    deltaf = 2*(dim1+1)
    b = deltaf/FLAGS.epsilon
    noise = np.random.laplace(loc=0.0, scale=b, size=data_len)
    # print(noise[:10])
    noise = noise.reshape([data_len, 1, 1])
    noise = np.ones([data_len, dim1+1, FLAGS.nb_labels]) * noise/((dim1+1)*FLAGS.nb_labels)
    return noise

def preprocess_data():
    x_train, y_train, x_test, y_test = load_data(FLAGS.dataset)
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    enc = OneHotEncoder()
    enc.fit(np.arange(FLAGS.nb_labels).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()
    dim1 = FLAGS.dim
    train_coding_path = "./train_dir/mnist_train.npy"
    test_coding_path = "./train_dir/mnist_test.npy"
    if os.path.exists(train_coding_path) and FLAGS.epochs==0:
        e_train = np.load(train_coding_path)
        e_test = np.load(test_coding_path)
    else:
        assert FLAGS.epochs > 0
        e_train, e_test = train_ae(x_train, x_test, dim1)
        np.save(train_coding_path, e_train)
        np.save(test_coding_path, e_test)
    noise = 
