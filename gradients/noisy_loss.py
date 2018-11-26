# coding: utf-8

from __future__ import division, print_function, unicode_literals, absolute_import

import tensorflow as tf 
import numpy as np 
import math

def compute_loss(a, t, graph, name=None):
    '''
    a: a=hw+b, is the log likelihood of data sample, shape=[batch_size, nb_labels]
    t: one-hot vector, ground truth label. shape=[batch_size, nb_labels]
    '''
    with tf.name_scope(name, "ComputeLoss", [a, t]) as name:
        return py_func(forward_func,
                                [a, t],
                                [tf.float32],
                                graph,
                                name=name,
                                grad=backprop_func)

def py_func(func, inp, Tout, graph, stateful=True, name=None, grad=None):
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

    tf.RegisterGradient(rnd_name)(grad)
    with graph.gradient_override_map({"PyFunc": rnd_name}):
        loss = tf.py_func(func, inp, Tout, stateful=stateful, name=name)[0]
        loss.set_shape([])
        return loss

def sigmoid(a):
    s = 1 / (1+np.exp(-a))
    return s

def forward_func(a, t):
    '''
    loss func doesn't matter the value of our method
    '''
    return np.maximum(a, 0) - np.multiply(a, t) + np.log(1 + np.exp(-np.abs(a)))
def forward_func2(a, t):
    '''
    compute binary cross entropy loss, refer to theano
    https://github.com/Theano/theano/blob/d395439aec5a6ddde8ef5c266fd976412a5c5695/theano/tensor/nnet/nnet.py#L2008-L2025
    '''
    s = sigmoid(a)
    logs = np.log(s)
    logs2 = np.log(1 - s)
    loss = np.multiply(t, logs) + np.multiply(1-t, logs2)
    return -loss

def backprop_func(op, grad):
    '''
    compute binary cross entropy loss's gradient over logit
    https://stats.stackexchange.com/questions/233499/neural-networks-bounded-output
    '''
    a = op.inputs[0]
    t = op.inputs[1]
    # t = tf.clip_by_value(t, 0.0, 1.0)
    # s = sigmoid(a)
    s = tf.nn.sigmoid(a)
    g = s - t
    zeros = tf.zeros_like(t)
    return g * grad, zeros
