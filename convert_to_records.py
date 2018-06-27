"""An example of how to convert to TFRecords."""

import os
import numpy as np
import tensorflow as tf

dirpath = '/home/joe/data/mnist'

raw_train_data = np.fromfile(os.path.join(dirpath, 'train-images-idx3-ubyte'), dtype=np.uint8)
raw_train_labels = np.fromfile(os.path.join(dirpath, 'train-labels-idx1-ubyte'), dtype=np.uint8)
raw_test_data = np.fromfile(os.path.join(dirpath, 't10k-images-idx3-ubyte'), dtype=np.uint8)
raw_test_labels = np.fromfile(os.path.join(dirpath, 't10k-labels-idx1-ubyte'), dtype=np.uint8)

train_data = np.reshape(raw_train_data[16:], (-1, 28, 28)).astype(float)
test_data = np.reshape(raw_test_data[16:], (-1, 28, 28)).astype(float)

train_labels = raw_train_labels[8:].astype(int)
test_labels = raw_test_labels[8:].astype(int)


def make_example(im, label):
  '''Convert a numpy array and label to a Tensorflow `Example`.'''

  im_feature = tf.train.Feature(float_list=tf.train.FloatList(
      value=im.reshape(-1)))
  label_feature = tf.train.Feature(int64_list=tf.train.Int64List(
      value=[label]))

  feature_dict = {
      'im': im_feature,
      'label': label_feature,
  }

  return tf.train.Example(features=tf.train.Features(feature=feature_dict))


with tf.python_io.TFRecordWriter(os.path.join(dirpath, 'mnist_train.tfrecords')) as writer:
  for i in range(len(train_data)):
    im = train_data[i]
    label = train_labels[i]
    example = make_example(im, label)
    writer.write(example.SerializeToString())


with tf.python_io.TFRecordWriter(os.path.join(dirpath, 'mnist_test.tfrecords')) as writer:
  for i in range(len(test_data)):
    im = test_data[i]
    label = test_labels[i]
    example = make_example(im, label)
    writer.write(example.SerializeToString())
