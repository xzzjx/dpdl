# coding: utf-8

from __future__ import division, print_function, unicode_literals, absolute_import
import tensorflow as tf 
import numpy as np 
import math

FLAGS = tf.flags.FLAGS

def get_weights(name, shape, stddev):
    weights = tf.get_variable(name, shape=shape, initializer=tf.truncated_normal_initializer(stddev=stddev))
    return weights
def get_biases(name, shape, init):
    biases = tf.get_variable(name, shape=shape, initializer=tf.constant_initializer(init))
    return biases

def max_out(inputs, num_units, axis=None):
    '''
    inputs has interger multiples units of output
    num_units is the number of output
    max pool
    '''
    shape = inputs.get_shape().as_list()
    if shape[0] is None:
        shape[0] = -1
    if axis is None:  # Assume that channel is the last dimension
        axis = -1
    num_channels = shape[axis]
    if num_channels % num_units:
        raise ValueError('number of features({}) is not '
                            'a multiple of num_units({})'.format(num_channels, num_units))
    shape[axis] = num_units
    shape += [num_channels // num_units]
    outputs = tf.reduce_max(tf.reshape(inputs, shape), -1, keep_dims=False)
    return outputs

def inference(images):

    first_conv_shape = [5, 5, 1, 20]

    with tf.variable_scope('conv1') as scope:
        kernel = get_weights('kernel', shape=first_conv_shape, stddev=1e-4)
        conv = tf.nn.conv2d(images, kernel, strides=[1, 1, 1, 1], padding='SAME')
        biases = get_biases('biases', shape=[20], init=0.0)
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope.name)
    
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    with tf.variable_scope('conv2') as scope:
        kernel = get_weights('kernel', shape=[5, 5, 20, 50], stddev=1e-4)
        conv = tf.nn.conv2d(pool1, kernel, strides=[1,1,1,1], padding='SAME')
        biases = get_biases('biases', shape=[50], init=0.1)
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope.name)
    pool2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    fc1 = tf.contrib.layers.flatten(pool2)
    with tf.variable_scope('fc1') as scope:
        # fc1 = tf.reshape(pool2, [FLAGS.batch_size, -1])
        # print(fc1.get_shape())
        dim = fc1.get_shape()[1]
        weights = get_weights('weights', shape=[dim, 500], stddev=0.04)
        biases = get_biases('biases', shape=[500], init=0.0)
        fc1 = tf.matmul(fc1, weights) + biases
    
        fc1 = tf.nn.relu(fc1, name=scope.name)


    with tf.variable_scope('fc3') as scope:
        weights = get_weights('weights', shape=[500, 10], stddev=0.04)
        biases = get_biases('biases', shape=[10], init=0.0)
        fc3 = tf.matmul(fc1, weights) + biases
        # fc3 = fc3 * noise + fc3
        # fc3 = tf.nn.relu(hfc1, name=scope.name)
    
    return fc3

def loss_fun(logits, y):
    '''
    y is one-hot vector
    '''
    zeros = tf.zeros_like(logits, dtype=logits.dtype)
    cond = (logits >= zeros)
    relu_logits = tf.where(cond, logits, zeros)
    neg_abs_logits = tf.where(cond, -logits, logits)

    loss = tf.add(relu_logits - logits * y, tf.log(1 + tf.exp(neg_abs_logits)))
    return loss