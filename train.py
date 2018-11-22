# coding: utf-8

from __future__ import division, print_function, unicode_literals, absolute_import
import tensorflow as tf 
import numpy as np 
import time
from datetime import datetime
import utils 
import mlenet
import svhn
import math
import input
from sklearn.preprocessing import OneHotEncoder

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('batch_size', 1800, 'batch_size')
# tf.flags.DEFINE_integer('hk', 25, 'number of top hidden layer neurons')
tf.flags.DEFINE_integer('stdnt_share', 5000, 'student share')
tf.flags.DEFINE_integer('max_steps', 3000, 'max steps train students')
tf.flags.DEFINE_float('epsilon', 0.15, 'privacy epsilon')
tf.flags.DEFINE_float('delta', 1e-5, 'privacy delta')
tf.flags.DEFINE_float('label_ratio', 0.5, 'ratio of labeled data')
tf.flags.DEFINE_float('unlabel', 0.1, 'label value for missed')
tf.flags.DEFINE_integer('nb_labels', 10, 'number of dataset labels')
tf.flags.DEFINE_string('dataset', 'mnist', 'dataset name, mnist, svhn, or cifar10')
tf.flags.DEFINE_string('data_dir', './data_dir', 'file dir path to store data')

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

def perturb(train_labels):
    '''
    train_labels are one-hot vector
    uniform sampling label and set it to unlabel value
    '''
    mask = np.random.uniform(size=train_labels.shape[0])
    null = np.ones(shape=[10], dtype=np.float32) * 0.1
    # train_labels = np.where(mask > 0.9, null, train_labels)
    train_labels[mask > FLAGS.label_ratio] = null
    return train_labels

def get_noise_lable(train_labels):
    '''
    train_labels are one-hot vector
    '''
    deltaf = FLAGS.nb_labels * 2
    epsilon = FLAGS.epsilon
    # batch_size = FLAGS.batch_size
    # scale = deltaf / (epsilon * batch_size)
    datalen = train_labels.shape[0]
    scale = deltaf / (epsilon * datalen)
    noise = np.random.laplace(0.0, scale, FLAGS.nb_labels)
    noise = np.reshape(noise, [FLAGS.nb_labels])
    train_labels = train_labels + noise
    train_labels = (1-FLAGS.label_ratio)/10 + FLAGS.label_ratio*train_labels
    return train_labels
def train_with_noise_ce(train_data, train_labels, ckpt_path):
    tf.reset_default_graph()
    with tf.Graph().as_default() as g:
        train_data_shape = train_data.shape
        # train_data_node = tf.placeholder(dtype=tf.float32, shape=[None, train_data_shape], name='train_data_node')
        train_data_node = tf.placeholder(dtype=tf.float32, shape=[None, train_data_shape[1], train_data_shape[2], train_data_shape[3]], name='train_data_node')
        train_labels_node = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='test_labels_node')
        
        print('placeholder done')
        # logits = fc.inference(train_data_node)
        # loss = fc.loss_fun(logits, train_labels_node)
        if FLAGS.dataset == 'mnist':
            logits = mlenet.inference(train_data_node)
            loss = mlenet.loss_fun(logits, train_labels_node)
        elif FLAGS.dataset == 'svhn':
            logits = svhn.inference(train_data_node)
            loss = svhn.loss_fun(logits, train_labels_node)
        # print(loss.get_shape())
        op = tf.train.AdamOptimizer(learning_rate=5e-4, beta1=0.9, beta2=0.999, name="student_op").minimize(loss)
        saver = tf.train.Saver(tf.global_variables())
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        data_length = len(train_data)
        for step in xrange(FLAGS.max_steps):
            start_time = time.time()
            batch_indices = utils.random_batch_indices(data_length, FLAGS.batch_size)

            feed_dict = {train_data_node: train_data[batch_indices],
                                train_labels_node: train_labels[batch_indices]}
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
    return True

def softmax_preds(images, ckpt_path, return_logits=False):
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
#   train_data_node = tf.placeholder(dtype=tf.float32, shape=[None, images.shape[-1]])
  train_data_node = tf.placeholder(dtype=tf.float32, shape=[None, images.shape[1], images.shape[2], images.shape[3]])

  # Build a Graph that computes the logits predictions from the placeholder
#   logits = fc.inference(train_data_node)
  if FLAGS.dataset == 'mnist':
    logits = mlenet.inference(train_data_node)
  elif FLAGS.dataset == 'svhn':
    logits = svhn.inference(train_data_node)
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

def train():
    ckpt_path = './train_dir/mnist.ckpt'
    train_data, train_labels, test_data, test_labels = load_data(FLAGS.dataset)
    # train_data = np.pad(train_data, ((0, 0), (2,2), (2,2), (0,0)), 'constant')
    # test_data = np.pad(test_data, ((0, 0), (2,2), (2,2), (0,0)), 'constant')
    enc = OneHotEncoder()
    enc.fit(np.arange(FLAGS.nb_labels).reshape(-1, 1))
    train_labels = enc.transform(train_labels.reshape(-1, 1)).toarray()
    test_labels = enc.transform(test_labels.reshape(-1, 1)).toarray()
    # train_data, train_labels, test_data, test_labels = preprocessing_img()

    # train_data = train_data.reshape(train_data.shape[0], -1)
    # test_data = test_data.reshape(test_data.shape[0], -1)
    train_labels = perturb(train_labels)
    train_labels = get_noise_lable(train_labels)

    assert train_with_noise_ce(train_data, train_labels, ckpt_path)

    ckpt_path_final = ckpt_path + '-' + str(FLAGS.max_steps-1)
    logits = softmax_preds(train_data, ckpt_path_final)
    accuracy = np.sum(np.argmax(logits, -1) == np.argmax(train_labels, -1)) / len(train_labels)
    print("student's train accuracy is ", accuracy)
    logits = softmax_preds(test_data, ckpt_path_final)
    accuracy = np.sum(np.argmax(logits, -1) == np.argmax(test_labels, -1)) / len(test_labels)
    print("student's test accuracy is ", accuracy)
    return True

def main(argv=None):
    assert train()

if __name__ == '__main__':
    main()