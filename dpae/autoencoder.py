# coding: utf-8

from __future__ import division, print_function, unicode_literals, absolute_import
from keras.layers import Input, Dense 
from keras.models import Model
from dpdl import input, utils
import tensorflow as tf 
import numpy as np 
import time
import math
from datetime import datetime
from noisy_loss import compute_loss
from sklearn.preprocessing import OneHotEncoder
import os
import keras


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
tf.flags.DEFINE_float('label_ratio', 0.5, 'ratio of labeled data')




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

def autoencoder(dim, encoding_dim):
    input_img = Input(shape=(dim,))
    encoded = Dense(encoding_dim, activation='relu')(input_img)

    decoded = Dense(dim, activation='sigmoid')(encoded)
    autoencoder = Model(input_img, decoded)

    encoder = Model(input_img, encoded)
    encoded_input = Input(shape=(encoding_dim, ))
    decoder_layer = autoencoder.layers[-1]
    decoder = Model(encoded_input, decoder_layer(encoded_input))

    optimizer = keras.optimizers.Adam(lr=0.01)
    # autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    autoencoder.compile(optimizer=optimizer, loss='binary_crossentropy')
    
    return encoder, decoder, autoencoder

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
def get_weights(name, shape, stddev):
    # with tf.variable_scope("weights"):
    weights = tf.get_variable(name, shape=shape, initializer=tf.truncated_normal_initializer(stddev=stddev))
    return weights


def MLP(dim, train_data, train_labels, ckpt_path):
    # tf.reset_default_graph()
    x = tf.placeholder(dtype=tf.float32, shape=(None, dim+1), name='x')
    y = tf.placeholder(dtype=tf.float32, shape=(None, 10), name="y")
    w = get_weights("weight", shape=[dim+1, FLAGS.nb_labels], stddev=0.001)
    noise_node = tf.placeholder(dtype=tf.float32, shape=(dim+1, FLAGS.nb_labels), name='noise')
    loss = compute_loss(x, y, w, noise_node, tf.get_default_graph())
    # logit = tf.matmul(x, w)
    # loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logit)
    op = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(loss)
    
    saver = tf.train.Saver(tf.global_variables())
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    data_length = len(train_data)
    print(data_length)
    batches_per_epoch = train_data.shape[0] // FLAGS.batch_size
    noise = get_noise_batch(batches_per_epoch, dim)
    for step in xrange(FLAGS.max_steps):
        start_time = time.time()
        # batch_indices = utils.random_batch_indices(data_length, FLAGS.batch_size)
        batch_idx = step % batches_per_epoch
        start, end = utils.batch_indices_isolate(batch_idx, data_length, FLAGS.batch_size)
        feed_dict = {x: train_data[start: end, :],
                            y: train_labels[start: end, :],
                            noise_node: noise[batch_idx]}
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

def plot(imgs, decoder, o_imgs):
    import matplotlib.pyplot as plt
    
    decoded_imgs = decoder.predict(imgs)
    n = 10  # how many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(o_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

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

def get_noise_batch(batches_per_epoch, dim1):
    deltaf = 1
    b = deltaf/FLAGS.epsilon
    noise = np.random.laplace(loc=0.0, scale=b, size=batches_per_epoch*FLAGS.nb_labels*(dim1+1))
    print(noise[:10])
    noise = noise.reshape([batches_per_epoch, dim1+1, FLAGS.nb_labels])
    # noise = np.ones([data_len, dim1+1, FLAGS.nb_labels]) * noise/((dim1+1)*FLAGS.nb_labels)
    return noise

def get_noise_batch_em(batches_per_epoch, dim1):
    '''
    global sentivity: deltaf(th) = 1
    '''
    pass
def get_noise_seperate(data_len, dim1):
    deltaf = 1 # t*h <= 1
    b = deltaf / FLAGS.epsilon
    noise = np.random.laplace(loc=0.0, scale=b, size=(dim1 + 1) * FLAGS.nb_labels * data_len)
    noise = noise.reshape([data_len, dim1+1, FLAGS.nb_labels]) * 1 / FLAGS.batch_size
    return noise

def select_confidence(preds):
    # print(preds[:10])
    # preds_idx = preds.argsort()[-2:][::-1]
    confidences = []
    for i in range(preds.shape[0]):
        preds_i = preds[i]
        preds_idx_i = preds_i.argsort()[-2:][::-1]
        if preds_i[preds_idx_i[0]] > 0.9 and preds_i[preds_idx_i[1]] < 0.1:
            confidences.append(i)
    assert len(confidences) > 0
    return np.array(confidences)

def self_training(preds, e_train, y_train):
    confidence = select_confidence(preds)
    c_train = e_train[confidence]
    c_y = y_train[confidence]
    c_p = preds[confidence]
    # noise = noise[confidence]
    accuracy = np.sum(np.argmax(c_p, -1) == np.argmax(c_y, -1)) / len(c_y)
    print("student's confidence train accuracy is ", accuracy)
    print("student's confidence number is ", c_train.shape[0])
    # c_train_path = './train_dir/mnist_c_train.npy'
    # c_y_path = './train_dir/mnist_c_y.npy'
    # np.save(c_train_path, c_train)
    # np.save(c_y_path, c_y)
    confidence_idx_path = './train_dir/mnist_c_idx.npy'
    confidence_y_path = './train_dir/mnist_c_y.npy'
    np.save(confidence_idx_path, confidence)
    p_y = np.argmax(c_p, -1)
    enc = OneHotEncoder()
    enc.fit(np.arange(FLAGS.nb_labels).reshape(-1, 1))
    p_y = enc.transform(p_y.reshape(-1, 1)).toarray()
    np.save(confidence_y_path, p_y)
    return True

def mask_label(train_labels):
    '''
    train_labels are one-hot vector
    uniform sampling label and set it to unlabel value
    '''
    mask = np.random.uniform(size=train_labels.shape[0])
    null = np.ones(shape=[10], dtype=np.float32) * 0.0
    # train_labels = np.where(mask > 0.9, null, train_labels)
    train_labels[mask > FLAGS.label_ratio] = null
    return train_labels

def select_unlabel(data_len):
    # mask = np.random.uniform(size=data_len)
    # return mask < FLAGS.label_ratio
    mask = np.random.choice(data_len, int(data_len * FLAGS.label_ratio))
    return mask
def train():
    '''
    train multiple autoencoder
    '''
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
    # ae2.fit(encoded_imgs, encoded_imgs, epochs=50, batch_size=256, shuffle=True, validation_data=(x_))
    ckpt_path = './train_dir/mnist.ckpt'
    ckpt_path_final = ckpt_path + '-' + str(FLAGS.max_steps-1)
    e_train = np.concatenate([e_train, np.ones([e_train.shape[0], 1], dtype=np.float32)], axis=1)
    e_train = e_train / np.max(e_train, axis=-1, keepdims=True)
    e_test = np.concatenate([e_test, np.ones([e_test.shape[0], 1], dtype=np.float32)], axis=1)
    e_test = e_test / np.max(e_test, axis=-1, keepdims=True)
    # noise = get_noise(e_train.shape[0], dim1)
    # noise = get_noise_seperate(e_train.shape[0], dim1)
    # y_train = mask_label(y_train)
    mask = select_unlabel(e_train.shape[0])
    e_train_ = e_train[mask]
    y_train_ = y_train[mask]
    # y_ = np.argmax(y_train, axis=-1)
    # for i in range(10):
    #     print(np.sum(y_ == i))
    # noise_ = noise[mask]
    assert MLP(dim1, e_train_, y_train_, ckpt_path)
    train_logits = softmax_preds(dim1, e_train, ckpt_path_final)
    accuracy = np.sum(np.argmax(train_logits, -1) == np.argmax(y_train, -1)) / len(y_train)
    print("student's train accuracy is ", accuracy)
    test_logits = softmax_preds(dim1, e_test, ckpt_path_final)
    accuracy = np.sum(np.argmax(test_logits, -1) == np.argmax(y_test, -1)) / len(y_test)
    print("student's test accuracy is ", accuracy)
    assert self_training(train_logits, e_train, y_train)
    return True

def main(argv=None):
    assert train()

if __name__ == "__main__":
    main()