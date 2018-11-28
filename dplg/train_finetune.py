# coding: utf-8

from __future__ import division, print_function, unicode_literals, absolute_import
import tensorflow as tf 
import numpy as np 
import vgg16
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

def extract_feature(x_train, x_test):
    model = vgg16.buildvgg16()
    x_train = x_train.transpose((0, 3, 1, 2))
    x_test = x_test.transpose((0, 3, 1, 2))
    e_train = model.predict(x_train)
    e_test = model.predict(x_test)
    e_train = e_train.reshape((e_train.shape[0], -1))
    e_test = e_test.reshape((e_test.shape[0], -1))
    return e_train, e_test

