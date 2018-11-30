# coding: utf-8

from __future__ import division, print_function, unicode_literals, absolute_import
import numpy as np 
from keras.datasets import mnist
import keras
import keras.backend as K
from LeNet import LeNet
from keras.losses import categorical_crossentropy as logloss
import tensorflow as tf
import math
from mnist_cnn import Mnist_CNN

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('batch_size', 1800, 'batch_size')
tf.flags.DEFINE_integer('hk', 25, 'number of top hidden layer neurons')
tf.flags.DEFINE_integer('stdnt_share', 5000, 'student share')
tf.flags.DEFINE_integer('max_steps', 3000, 'max steps train students')
tf.flags.DEFINE_float('epsilon', 0.15, 'privacy epsilon')
tf.flags.DEFINE_float('delta', 1e-5, 'privacy delta')
tf.flags.DEFINE_float('label_ratio', 0.5, 'ratio of labeled data')

def preprocessing():
    (train_data, train_labels), (test_data, test_labels) = mnist.load_data()
    # train_data = K.expand_dims(train_labels, axis=-1)
    img_rows, img_cols = 28, 28
    train_data = train_data.reshape(train_data.shape[0], img_rows, img_cols, 1)
    test_data = test_data.reshape(test_data.shape[0], img_rows, img_cols, 1)
    train_data = train_data.astype('float32') / 255
    test_data = test_data.astype('float32') / 255
    # train_labels = keras.utils.to_categorical(train_labels, 10)
    test_labels = keras.utils.to_categorical(test_labels, 10)
    confidence_idx_path = './train_dir/mnist_c_idx.npy'
    confidence_y_path = './train_dir/mnist_c_y.npy'
    confidence = np.load(confidence_idx_path)
    train_data = train_data[confidence]
    true_y = train_labels[confidence]
    train_labels = np.load(confidence_y_path)
    y_ = np.argmax(train_labels, -1)
    print(np.sum(true_y == y_) / len(true_y))
    class_weight = {}
    mu = 1
    for i in range(10):
        # print(np.sum(y_ == i))
        class_weight[i] = mu*train_labels.shape[0]/np.sum(y_ == i)
        print(class_weight[i])
    return train_data[:10000], train_labels[:10000], test_data, test_labels, class_weight


def build_lenet():
    train_data, train_labels, test_data, test_labels, class_weight = preprocessing()
    print(train_data.shape)
    op = keras.optimizers.Adam()
    model = LeNet.build(width=28, height=28, depth=1, classes=10)
    # model.compile(loss='binary_crossentropy', optimizer=op, metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer=op, metrics=['accuracy'])

    print("[INFO] training...")
    model.fit(train_data, train_labels, batch_size=128, nb_epoch=5, verbose=1, class_weight=class_weight)

    print("[INFO] evaluating...")
    (loss, accuracy) = model.evaluate(test_data, test_labels, batch_size=128, verbose=1)
    print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))

    weightsPath = './train_dir/LeNet.hdf5'
    print("[INFO] dumping weights to file...")
    model.save_weights(weightsPath, overwrite=True)

def cnn():
    train_data, train_labels, test_data, test_labels, class_weight = preprocessing()
    print(train_data.shape)
    op = keras.optimizers.Adam()
    model = Mnist_CNN.build(input_shape=(28, 28, 1), num_classes=10)
    # binary crossentropy performs bettern than categorical crossentropy on validation data
    model.compile(loss='binary_crossentropy', optimizer=op, metrics=['accuracy'])
    # model.compile(loss='categorical_crossentropy', optimizer=op, metrics=['accuracy'])

    print("[INFO] training...")
    model.fit(train_data, train_labels, batch_size=128, nb_epoch=5, verbose=1, class_weight='auto', \
                    validation_data=(test_data, test_labels))

    print("[INFO] evaluating...")
    (loss, accuracy) = model.evaluate(test_data, test_labels, batch_size=128, verbose=1)
    print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))

    # weightsPath = './train_dir/LeNet.hdf5'
    # print("[INFO] dumping weights to file...")
    # model.save_weights(weightsPath, overwrite=True)



def main(argv=None):
    # build_lenet()
    cnn()
if __name__ == '__main__':
    main()
    # distillation()
    # build_lenet()