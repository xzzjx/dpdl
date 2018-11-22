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

def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  var = _variable_on_cpu(name, shape,
                         tf.truncated_normal_initializer(stddev=stddev))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def inference(images, dropout=True):
  """Build the CNN model.
  Args:
    images: Images returned from distorted_inputs() or inputs().
    dropout: Boolean controlling whether to use dropout or not
  Returns:
    Logits
  """
  if FLAGS.dataset == 'mnist':
    first_conv_shape = [5, 5, 1, 64]
  else:
    first_conv_shape = [5, 5, 3, 64]

  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = get_weights('kernel', first_conv_shape, 1e-4)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    # biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    biases = get_biases('biases', [64], 0.0)
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope.name)
    # if dropout:
      # conv1 = tf.nn.dropout(conv1, 0.3, seed=42)


  # pool1
  pool1 = tf.nn.max_pool(conv1,
                         ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1],
                         padding='SAME',
                         name='pool1')

  # norm1
  norm1 = tf.nn.lrn(pool1,
                    4,
                    bias=1.0,
                    alpha=0.001 / 9.0,
                    beta=0.75,
                    name='norm1')

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = get_weights('weights', shape=[5, 5, 64, 128], stddev=1e-4)
    # kernel = _variable_with_weight_decay('weights',
    #                                      shape=[5, 5, 64, 128],
    #                                      stddev=1e-4,
    #                                      wd=0.0)
    conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
    # biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.1))
    biases = get_biases('biases', [128], 0.1)
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name=scope.name)
    # if dropout:
      #conv2 = tf.nn.dropout(conv2, 0.3, seed=42)


  # norm2
  norm2 = tf.nn.lrn(conv2,
                    4,
                    bias=1.0,
                    alpha=0.001 / 9.0,
                    beta=0.75,
                    name='norm2')

  # pool2
  pool2 = tf.nn.max_pool(norm2,
                         ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1],
                         padding='SAME',
                         name='pool2')

  # local3
  with tf.variable_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    # reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
    reshape = tf.contrib.layers.flatten(pool2)
    dim = reshape.get_shape()[1].value
    weights = get_weights('weights', shape=[dim, 384], stddev=0.04)
    # weights = _variable_with_weight_decay('weights',
    #                                       shape=[dim, 384],
    #                                       stddev=0.04,
    #                                       wd=0.004)
    # biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    biases = get_biases('biases', [384], 0.1)
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    if dropout:
      local3 = tf.nn.dropout(local3, 0.5, seed=42)

  # local4
  with tf.variable_scope('local4') as scope:
    # weights = _variable_with_weight_decay('weights',
    #                                       shape=[384, 192],
    #                                       stddev=0.04,
    #                                       wd=0.004)
    weights = get_weights('weights', shape=[384, 192], stddev=0.04)
    # biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    biases = get_biases('biases', [192], 0.1)
    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
    if dropout:
      local4 = tf.nn.dropout(local4, 0.5, seed=42)

  # compute logits
  with tf.variable_scope('softmax_linear') as scope:
    # weights = _variable_with_weight_decay('weights',
    #                                       [192, FLAGS.nb_labels],
    #                                       stddev=1/192.0,
    #                                       wd=0.0)
    weights = get_weights('weights', shape=[192, FLAGS.nb_labels], stddev=1/192.0)
    # biases = _variable_on_cpu('biases',
    #                           [FLAGS.nb_labels],
    #                           tf.constant_initializer(0.0))
    biases = get_biases('biases', [FLAGS.nb_labels], 0.0)
    logits = tf.add(tf.matmul(local4, weights), biases, name=scope.name)

  return logits

def loss_fun(logits, y):
    '''
    y is one-hot vector
    '''
    zeros = tf.zeros_like(logits, dtype=logits.dtype)
    cond = (logits >= zeros)
    relu_logits = tf.where(cond, logits, zeros)
    neg_abs_logits = tf.where(cond, -logits, logits)

    # hk = 25
    # deltaf = 10 * 2
    # epsilon = FLAGS.epsilon
    # batch_size = FLAGS.batch_size
    # scale = deltaf / (epsilon * batch_size)
    # noise = np.random.laplace(0.0, scale, 10)
    # noise = np.reshape(noise, [10])
    # y = y + noise
    # y = (1-FLAGS.label_ratio)/10 + FLAGS.label_ratio*y
    # loss = tf.add(relu_logits - logits * y, math.log(2.0) + 0.5*neg_abs_logits + 1.0 / 8.0 * neg_abs_logits**2, name='noise_loss')
    loss = tf.add(relu_logits - logits * y, tf.log(1 + tf.exp(neg_abs_logits)))
    return loss
