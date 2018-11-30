# coding: utf-8

from __future__ import division, print_function, unicode_literals, absolute_import

import tensorflow as tf 
import numpy as np 
import math

FLAGS = tf.flags.FLAGS

def compute_loss(a, t, w, n, graph, name=None):
    '''
    a: a=hw+b, is the log likelihood of data sample, shape=[batch_size, nb_labels]
    t: one-hot vector, ground truth label. shape=[batch_size, nb_labels]
    '''
    with tf.name_scope(name, "ComputeLoss", [a, t]) as name:
        return py_func(forward_func,
                                [a, t, w, n],
                                [tf.float32],
                                graph,
                                name=name,
                                # grad=backprop_func)
                                grad=backprop_func_noise)

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

def forward_func(a, t, w, n):
    '''
    loss func doesn't matter the value of our method
    '''
    a = np.dot(a, w)
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
    a = op.inputs[0]
    t = op.inputs[1]
    w = op.inputs[2]
    ag = tf.zeros_like(a)
    tg = tf.zeros_like(t)
    print('a.shape: ', a.get_shape())
    print('t.shape: ', t.get_shape())
    print('w.shape: ', w.get_shape())
    
    logit = tf.matmul(a, w, name="aw_mul") #(None, 10)
    logit = tf.expand_dims(logit, axis=1)
    a = tf.expand_dims(a, axis=2)
    t = tf.expand_dims(t, axis=1)
    wg = tf.reduce_mean(a*tf.nn.sigmoid(logit) - a*t, axis=0)

    return ag, tg, wg*grad

def backprop_func_noise(op, grad):
    a = op.inputs[0]
    t = op.inputs[1]
    w = op.inputs[2]
    noise = op.inputs[3]
    ag = tf.zeros_like(a)
    tg = tf.zeros_like(t)
    ng = tf.zeros_like(noise)
    print('a.shape: ', a.get_shape())
    print('t.shape: ', t.get_shape())
    print('w.shape: ', w.get_shape())
    
    logit = tf.matmul(a, w, name="aw_mul") #(None, 10)
    logit = tf.expand_dims(logit, axis=1)
    a = tf.expand_dims(a, axis=2)
    t = tf.expand_dims(t, axis=1)

    a = tf.Print(a, [a], message='a: ', first_n=10, summarize=100)
    t = tf.Print(t, [t], message='t: ', first_n=10, summarize=100)
    noise = 1 / FLAGS.batch_size * noise
    # noise_at = noise + a*t
    noise_at = tf.reduce_mean(a*t, axis=0) + noise
    noise_at = tf.clip_by_value(noise_at, -1.0, 1.0)
    noise_at = tf.Print(noise_at, [noise_at], message='noise_at: ', first_n=10, summarize=100)
    wg_noise = tf.reduce_mean(a*tf.nn.sigmoid(logit), axis=0) - noise_at
    # wg = tf.reduce_mean(a*tf.nn.sigmoid(logit) - noise_at, axis=0)
    
    # wg = tf.reduce_mean(a*tf.nn.sigmoid(logit) - a*t, axis=0)
    # wg_noise = wg + tf.reduce_mean(noise, axis=0)
    # wg_noise = wg + noise

    # wg_noise = tf.clip_by_value(wg_noise, -1.0, 1.0)

    return ag, tg, wg_noise*grad, ng