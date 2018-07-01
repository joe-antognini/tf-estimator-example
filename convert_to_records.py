#! /usr/bin/env python

import argparse
import gzip
import os
import numpy as np
import urllib
import tensorflow as tf

MNIST_URL = 'http://yann.lecun.com/exdb/mnist/'
MNIST_FNAMES = ['train-images-idx3-ubyte.gz',
                'train-labels-idx1-ubyte.gz',
                't10k-images-idx3-ubyte.gz',
                't10k-labels-idx1-ubyte.gz']


def download_mnist(data_dir):
  """Download the MNIST data set.

  Args:
    data_dir: Directory to save the data to.
  """
  for fname in MNIST_FNAMES:
    urllib.urlretrieve(MNIST_URL + fname, os.path.join(data_dir, fname))


def convert_mnist_to_numpy(data_dir):
  """Convert the MNIST format to Numpy and save to disk."""
  for fname in MNIST_FNAMES:
    with gzip.open(os.path.join(data_dir, fname), 'rb') as infile:
      raw_file_content = infile.read()
    file_content = np.fromstring(raw_file_content, dtype=np.uint8)
    if 'images' in fname:
      arr = np.reshape(file_content[16:], (-1, 28, 28)).astype(float)
    elif 'labels' in fname:
      arr = file_content[8:].astype(int)
    outname = '_'.join(fname.split('-')[:2])
    if outname.startswith('t10k'):
      outname = 'test' + outname[4:]

    np.save(os.path.join(data_dir, outname), arr)
      

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


def convert_numpy_to_tfrecords(data_dir):
  """Convert the Numpy format to TFRecords and save to disk."""
  for mode in ['train', 'test']:
    images = np.load(os.path.join(data_dir, mode + '_images.npy'))
    labels = np.load(os.path.join(data_dir, mode + '_labels.npy'))

    output_fname = os.path.join(data_dir, mode + '.tfrecords')
    with tf.python_io.TFRecordWriter(output_fname) as writer:
      for im, label in zip(images, labels):
        example = make_example(im, label)
        writer.write(example.SerializeToString())


def clean_up(data_dir):
  """Remove the gzip files."""
  for fname in MNIST_FNAMES:
    os.remove(os.path.join(data_dir, fname))


def main(args):
  download_mnist(args.data_dir)
  convert_mnist_to_numpy(args.data_dir)
  convert_numpy_to_tfrecords(args.data_dir)
  clean_up(args.data_dir)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Convert MNIST data to TFRecords')
  parser.add_argument('--data_dir')
  args = parser.parse_args()
  main(args)
