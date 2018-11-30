# coding: utf-8

from __future__ import division, print_function, unicode_literals, absolute_import
import tensorflow as tf 
import numpy as np 
import math
from dpdl import input, utils
from keras.layers import Input, Dense 
from keras.models import Model
from sklearn.preprocessing import OneHotEncoder
import time 
from datetime import datetime
from noisy_loss import compute_loss
import os


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
tf.flags.DEFINE_integer('nb_classifiers', 2, 'num of classifiers')
tf.flags.DEFINE_float('sample_ratio', 0.5, 'sample ratio of bootstrap')



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
    noise = 1/FLAGS.batch_size * np.random.laplace(loc=0.0, scale=b, size=data_len)
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
    noise = get_noise(e_train.shape[0], dim1)

    return e_train, y_train, e_test, y_test, noise

def bootstrap(data_len, q):
    random_indices = np.arange(data_len, dtype=np.int32)
    np.random.shuffle(random_indices)
    start = 0
    end = int(q * data_len)
    return random_indices[start: end]

def get_weights(name, shape, stddev):
    # with tf.variable_scope("weights"):
    weights = tf.get_variable(name, shape=shape, initializer=tf.truncated_normal_initializer(stddev=stddev))
    return weights

def MLP(dim, train_data, train_labels, noise, ckpt_path):
    # tf.reset_default_graph()
    x = tf.placeholder(dtype=tf.float32, shape=(None, dim+1), name='x')
    y = tf.placeholder(dtype=tf.float32, shape=(None, 10), name="y")
    w = get_weights("weight", shape=[dim+1, FLAGS.nb_labels], stddev=0.001)
    noise_node = tf.placeholder(dtype=tf.float32, shape=(None, dim+1, FLAGS.nb_labels), name='noise')
    loss = compute_loss(x, y, w, noise_node, tf.get_default_graph())
    # logit = tf.matmul(x, w)
    # loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logit)
    op = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(loss)
    
    saver = tf.train.Saver(tf.global_variables())
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    data_length = len(train_data)
    for step in xrange(FLAGS.max_steps):
        start_time = time.time()
        batch_indices = utils.random_batch_indices(data_length, FLAGS.batch_size)

        feed_dict = {x: train_data[batch_indices],
                            y: train_labels[batch_indices],
                            noise_node: noise[batch_indices]}
        _, loss_value= sess.run([op, loss], feed_dict = feed_dict)

        duration = time.time() - start_time

        if step % 100 == 0:
            num_examples_per_step = FLAGS.batch_size 
            examples_per_sec = num_examples_per_step / duration
            sec_per_batch = float(duration)

            format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')

            print(format_str % (datetime.now(), step, np.mean(loss_value), examples_per_sec, sec_per_batch))

        if step % 1000 == 0 or (step+1) == FLAGS.max_steps:
            saver.save(sess, ckpt_path, global_step=step)
    tf.reset_default_graph()
    return True

def inference(dim, train_data):
    # x = tf.placeholder(dtype=tf.float32, shape=(None, dim+1), name='x')
    # y = tf.placeholder(dtype=tf.float32, shape=(None,), name="y")
    # with tf.variable_scope("weights", reuse=False):
    w = get_weights("weight", shape=[dim+1, FLAGS.nb_labels], stddev=0.001)
    return tf.matmul(train_data, w)


def softmax_preds(dim, images, ckpt_path, return_logits=False):
  """
  Compute softmax activations (probabilities) with the model saved in the path
  specified as an argument
  :param images: a np array of images
  :param ckpt_path: a TF model checkpoint
  :param logits: if set to True, return logits instead of probabilities
  :return: probabilities (or logits if logits is set to True)
  """
  # Compute nb samples and deduce nb of batches
  data_length = len(images)
  nb_batches = math.ceil(len(images) / FLAGS.batch_size)

  # Declare data placeholder
  train_data_node = tf.placeholder(dtype=tf.float32, shape=[None, dim+1])
#   train_data_node = tf.placeholder(dtype=tf.float32, shape=[None, images.shape[1], images.shape[2], images.shape[3]])

  # Build a Graph that computes the logits predictions from the placeholder
#   logits = fc.inference(train_data_node)
  if FLAGS.dataset == 'mnist':
    logits = inference(dim, train_data_node)
  elif FLAGS.dataset == 'svhn':
    logits = inference(dim, train_data_node)
    # logits = inference2(train_data_node)

  if return_logits:
    # We are returning the logits directly (no need to apply softmax)
    output = logits
  else:
    # Add softmax predictions to graph: will return probabilities
    output = tf.nn.softmax(logits)

  # Restore the moving average version of the learned variables for eval.
  # variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
  # variables_to_restore = variable_averages.variables_to_restore()
  # saver = tf.train.Saver(variables_to_restore)
  saver = tf.train.Saver()

  # Will hold the result
  preds = np.zeros((data_length, 10), dtype=np.float32)

  # Create TF session
  with tf.Session() as sess:
    # Restore TF session from checkpoint file
    saver.restore(sess, ckpt_path)

    # Parse data by batch
    for batch_nb in xrange(0, int(nb_batches+1)):
      # Compute batch start and end indices
      start, end = utils.batch_indices(batch_nb, data_length, FLAGS.batch_size)

      # Prepare feed dictionary
      feed_dict = {train_data_node: images[start:end]}

      # Run session ([0] because run returns a batch with len 1st dim == 1)
      preds[start:end, :] = sess.run([output], feed_dict=feed_dict)[0]

  # Reset graph to allow multiple calls
  tf.reset_default_graph()

  return preds

def train(e_train, y_train, e_test, y_test, noise, ckpt_path):
    dim1 = FLAGS.dim
    ckpt_path_final = ckpt_path + '-' + str(FLAGS.max_steps-1)
    
    assert MLP(dim1, e_train, y_train, noise, ckpt_path)
    logits = softmax_preds(dim1, e_train, ckpt_path_final)
    accuracy = np.sum(np.argmax(logits, -1) == np.argmax(y_train, -1)) / len(y_train)
    print("student's train accuracy is ", accuracy)
    logits = softmax_preds(dim1, e_test, ckpt_path_final)
    accuracy = np.sum(np.argmax(logits, -1) == np.argmax(y_test, -1)) / len(y_test)
    print("student's test accuracy is ", accuracy)
    return True

def train_mul():
    ckpt_path_head = './train_dir/mnist'
    nb_classifiers = FLAGS.nb_classifiers

    e_train, y_train, e_test, y_test, noise = preprocess_data()
    e_train = np.concatenate([e_train, np.ones([e_train.shape[0], 1], dtype=np.float32)], axis=1)
    e_train = e_train / np.max(e_train)
    e_test = np.concatenate([e_test, np.ones([e_test.shape[0], 1], dtype=np.float32)], axis=1)
    e_test = e_test / np.max(e_test)
    noise = get_noise(e_train.shape[0], FLAGS.dim)
    predicts = []
    for i in range(nb_classifiers):
        ckpt_path = ckpt_path_head + str(i) + '.ckpt'
        indices = bootstrap(e_train.shape[0], FLAGS.sample_ratio)

        assert train(e_train[indices], y_train[indices], e_test, y_test, noise[indices], ckpt_path)

        ckpt_path_final = ckpt_path + '-' + str(FLAGS.max_steps-1)
        logits = softmax_preds(FLAGS.dim, e_train, ckpt_path_final)
        predicts.append(np.argmax(logits, -1))


    predicts = np.array(predicts)
    predicts = predicts.transpose(1, 0)
    predicts_file = './train_dir/mnist_predcits.npy'
    np.save(predicts_file, predicts)

    bins = []
    for i in range(predicts.shape[0]):
        bins.append(np.bincount(predicts[i], minlength=FLAGS.nb_labels))
    
    p_ = np.zeros(shape=[predicts.shape[0]], dtype=np.int32)
    for i in range(predicts.shape[0]):
        p_[i] = np.argmax(bins[i])
    
    accuracy = np.sum(p_ == np.argmax(y_train, -1)) / len(y_train)

    print("train accuracy of bagging is: ", accuracy)
    return True

def select_confident():
    pass

def main(argv=None):
    train_mul()

if __name__ == '__main__':
    main()