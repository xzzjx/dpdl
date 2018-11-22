#coding: utf-8

from __future__ import division, unicode_literals, print_function, absolute_import
import tensorflow as  tf 
import numpy as np 
from keras.layers import Dense, Activation
import keras
import math

FLAGS = tf.flags.FLAGS

class StudentModel:
    @staticmethod
    def build(input_shape, classes):

        # input_shape is 1-d vector
        model = keras.Sequential()

        model.add(Dense(800, input_dim=input_shape))
        model.add(Activation("relu"))
        model.add(Dense(800))
        model.add(Activation('relu'))
        model.add(Dense(classes))
        model.add(Activation('softmax'))

        return model
    
def weight_variable(name, shape, stddev):
    # initial = tf.truncated_normal(shape, stddev=stddev)
    weight = tf.get_variable(name, shape, initializer=tf.truncated_normal_initializer(stddev=stddev))
    return weight

def bias_variable(name, shape, initial):
    bias = tf.get_variable(name, shape, initializer=tf.constant_initializer(initial))
    return bias

def max_out(inputs, num_units, axis=None):
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
    img_row, img_col, depth = 28, 28, 1
    hk = FLAGS.hk
    with tf.variable_scope('layer1') as scope:
        weights = weight_variable('weight', shape=[img_row * img_col * depth, 100], stddev=1.0 / 500)
        biases = bias_variable('bias', [100], initial=0.1)
        layer1 = tf.nn.relu(tf.matmul(images, weights) + biases, name=scope.name)

    with tf.variable_scope('layer2') as scope:
        weights = weight_variable('weight', shape=[100, hk], stddev=1/100.0)
        biases = bias_variable('bias', [hk], initial=0.1)
        # layer2 = tf.nn.relu(tf.matmul(layer1, weights) + biases, name=scope.name)
        layer2 = tf.matmul(layer1, weights) + biases
    
    with tf.variable_scope('batch_norm') as scope:
        batch_mean, batch_var = tf.nn.moments(layer2, [0])
        beta2 = tf.zeros_like(layer2)
        scale2 = tf.ones_like(layer2)
        BN = tf.nn.batch_normalization(layer2, batch_mean, batch_var, beta2, scale2, 1e-3)

    with tf.variable_scope('FM') as scope:
        hfc1 = max_out(BN, hk)
        hfc1 = tf.clip_by_value(hfc1, -1, 1)
        deltaf = 10 * (hk + 1/4 * (hk ** 2))
        epsilon = FLAGS.epsilon
        batch_size = FLAGS.batch_size
        scale = deltaf / (epsilon * batch_size)
        noise = np.random.laplace(0.0, scale, hk)
        noise = np.reshape(noise, [hk])
        hfc1 = hfc1 + noise

        weights = weight_variable('weight', shape=[hk, 10], stddev=1.0/hk)
        biases = bias_variable('bias', [10], initial=0.1)
        logits = tf.matmul(hfc1, weights) + biases

    return logits

def loss_fun(logits, y):
    '''
    y is one-hot vector
    '''
    zeros = tf.zeros_like(logits, dtype=logits.dtype)
    cond = (logits >= zeros)
    relu_logits = tf.where(cond, logits, zeros)
    neg_abs_logits = tf.where(cond, -logits, logits)

    loss = tf.add(relu_logits - logits * y, math.log(2.0) + 0.5*neg_abs_logits + 1.0 / 8.0 * neg_abs_logits**2, name='noise_loss')
    return loss