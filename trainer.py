#! /usr/bin/env python

import argparse
import logging
import os
import time

import tensorflow as tf


N_LABELS = 10
N_EVAL_EXAMPLES = 10000
N_TRAIN_EXAMPLES = 60000


def build_input_fn(fnames, hparams, is_training=True, use_tpu=True):
  '''Build the input function.'''
  def parse_fn(proto):
    '''Parse a single Tensorflow example from a `TFRecord`.

    Args:
      proto: The serialized protobuf of the `tf.Example`

    Returns:
      A `Tensor` containing the image.
      A one-hot `Tensor` containing the label.
    '''

    features = {
        'im': tf.FixedLenFeature([28 * 28], tf.float32),
        'label': tf.FixedLenFeature([], tf.int64),
    }
    parsed_features = tf.parse_single_example(proto, features)
    im = tf.reshape(parsed_features['im'], [28, 28, 1])
    label = tf.one_hot(parsed_features['label'], N_LABELS)

    return im, label

  def input_fn(params):
    '''Feed input into the graph.'''
    with tf.variable_scope('image_preprocessing'):
      dataset = tf.data.TFRecordDataset(fnames)
      dataset = dataset.shuffle(len(fnames))
      dataset = dataset.map(parse_fn)
      if is_training:
        dataset = dataset.shuffle(args.shuffle_buffer_size)
        dataset = dataset.repeat()
      if use_tpu:
        dataset = dataset.apply(
            tf.contrib.data.batch_and_drop_remainder(params['batch_size']))
      else:
        dataset = dataset.batch(hparams.batch_size)
      dataset = dataset.prefetch(buffer_size=1)
      iterator = dataset.make_one_shot_iterator()
      features, label = iterator.get_next()

    return features, label
  return input_fn


def model_body(features):
  net = tf.identity(features)
  net = tf.layers.conv2d(net, filters=32, kernel_size=5, activation=tf.nn.relu)
  net = tf.layers.max_pooling2d(net, pool_size=2, strides=2)
  net = tf.layers.conv2d(net, filters=64, kernel_size=5, activation=tf.nn.relu)
  net = tf.layers.max_pooling2d(net, pool_size=2, strides=2)
  net = tf.contrib.layers.flatten(net)
  net = tf.layers.dense(net, 1024, activation=tf.nn.relu)

  return net


def build_model_fn(hparams):
  '''Build the model function.'''
  def model_fn(features, labels, mode, params=None):
    '''Define the model graph.'''

    if params:
      hparams.override_from_dict(params)

    net = model_body(features)

    logits = tf.layers.dense(net, units=N_LABELS)
    xentropies = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                         labels=labels)
    loss = tf.reduce_mean(xentropies)

    if hparams.get('learning_rate_decay_scheme') == 'exponential':
      learning_rate = tf.train.exponential_decay(hparams.learning_rate,
                                                 tf.train.get_global_step(),
                                                 hparams.decay_steps,
                                                 hparams.decay_rate)
    else:
      learning_rate = hparams.learning_rate

    optimizer = tf.train.MomentumOptimizer(learning_rate,
                                           hparams.momentum)

    if hparams.use_tpu:
      optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    def metric_fn(labels, logits):
      predictions = tf.argmax(logits, axis=1)
      accuracy = tf.metrics.accuracy(tf.argmax(labels, axis=1), predictions)
      return {'accuracy': accuracy}
    eval_metrics = (metric_fn, [labels, logits])
    eval_metric_ops = metric_fn(labels, logits)

    if hparams.use_tpu:
      return tf.contrib.tpu.TPUEstimatorSpec(mode=mode,
                                             loss=loss,
                                             train_op=train_op,
                                             eval_metrics=eval_metrics)
    else:
      return tf.estimator.EstimatorSpec(mode=mode,
                                        loss=loss,
                                        train_op=train_op,
                                        eval_metric_ops=eval_metric_ops)
  return model_fn


def main(args):
  eval_batch_size = args.eval_batch_size
  if args.mode == 'train':
    eval_batch_size = None

  hparams = tf.contrib.training.HParams(
      learning_rate=args.learning_rate,
      momentum=args.momentum,
      batch_size=args.batch_size,
      eval_batch_size=eval_batch_size,
      use_tpu=False)

  if hparams.use_tpu:
    print 'Using TPU!'
    run_config = tf.contrib.tpu.RunConfig(
        master=args.tpu_master,
        evaluation_master=args.tpu_master,
        model_dir=args.model_dir,
        save_checkpoints_steps=args.save_checkpoints_steps,
        save_summary_steps=args.save_summary_steps,
        keep_checkpoint_max=args.keep_checkpoint_max,
        session_config=tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=True),
        tpu_config=tf.contrib.tpu.TPUConfig(args.tpu_iterations_per_loop,
                                            args.tpu_num_shards))
    nn = tf.contrib.tpu.TPUEstimator(model_fn=build_model_fn(hparams),
                                     config=run_config,
                                     train_batch_size=hparams.batch_size,
                                     eval_batch_size=hparams.eval_batch_size)
  else:
    run_config = tf.estimator.RunConfig(
        model_dir=args.model_dir,
        save_checkpoints_steps=args.save_checkpoints_steps,
        save_summary_steps=args.save_summary_steps,
        keep_checkpoint_max=args.keep_checkpoint_max)
    nn = tf.estimator.Estimator(model_fn=build_model_fn(hparams),
                                config=run_config)

  train_fnames = [args.train_file_pattern]
  eval_fnames = [args.eval_file_pattern]
  logging.info('mode: {}'.format(args.mode))

  if args.mode == 'train':
    logging.info('Starting training.')
    input_fn = build_input_fn(train_fnames, hparams, is_training=True,
                              use_tpu=args.use_tpu)
    nn.train(input_fn=input_fn, steps=args.train_steps)
  elif args.mode == 'eval':
    while not tf.train.checkpoint_exists(os.path.join(args.model_dir, 'model')):
      logging.info(
          'No checkpoint found.  Waiting 60 seconds before trying again.')
      time.sleep(60)

    while True:
      input_fn = build_input_fn(eval_fnames, hparams, is_training=False,
                                use_tpu=args.use_tpu)
      nn.evaluate(input_fn=input_fn,
                  steps=int(N_EVAL_EXAMPLES / hparams.eval_batch_size))
  elif args.mode == 'train_and_eval':
    i_steps = 0

    # Get steps at initialization
    input_fn = build_input_fn(train_fnames, hparams, is_training=True,
                              use_tpu=args.use_tpu)
    nn.evaluate(input_fn=input_fn,
                steps=int(N_TRAIN_EXAMPLES / hparams.batch_size),
                name='train')
    input_fn = build_input_fn(eval_fnames, hparams, is_training=False,
                              use_tpu=args.use_tpu)
    nn.evaluate(input_fn=input_fn,
                steps=int(N_EVAL_EXAMPLES / hparams.batch_size),
                name='test')

    # Start training!
    while i_steps < args.train_steps:
      input_fn = build_input_fn(train_fnames, hparams, is_training=True,
                                use_tpu=args.use_tpu)
      nn.train(input_fn=input_fn, steps=args.eval_freq)
      nn.evaluate(input_fn=input_fn,
                  steps=int(N_TRAIN_EXAMPLES / hparams.batch_size),
                  name='train')
      input_fn = build_input_fn(eval_fnames, hparams, is_training=False,
                                use_tpu=args.use_tpu)
      nn.evaluate(input_fn=input_fn,
                  steps=int(N_EVAL_EXAMPLES / hparams.batch_size),
                  name='test')
      i_steps += args.eval_freq


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='Run an MNIST model.')
  parser.add_argument('--batch_size', type=int, default=128,
                      help='Batch size.')
  parser.add_argument('--eval_batch_size', type=int, default=128,
                      help='Batch size on eval.')
  parser.add_argument('--train_file_pattern',
                      help='File pattern of the training data.')
  parser.add_argument('--eval_file_pattern',
                      help='File pattern of the eval data.')
  parser.add_argument('--eval_freq', type=int, default=500,
                      help='Number of steps between eval runs.')
  parser.add_argument('--tpu_iterations_per_loop', type=int, default=500,
                      help='Number of iterations in a TPU cycle.')
  parser.add_argument('--keep_checkpoint_max', type=int, default=2,
                      help='Maximum number of checkpoints to keep.')
  parser.add_argument('--learning_rate', type=float, default=0.1,
                      help='Initial learning rate for the optimizer.')
  parser.add_argument('--mode', choices=['train', 'eval', 'train_and_eval'],
                      help='Which mode to run in.')
  parser.add_argument('--model_dir',
                      help='Directory to save model checkpoints.')
  parser.add_argument('--momentum', type=float, default=0.9,
                      help='Momentum for the optimizer.')
  parser.add_argument('--tpu_num_shards', type=int, default=8,
                      help='Number of TPU shards.')
  parser.add_argument('--save_checkpoints_steps', type=int, default=100,
                      help='Number of steps between checkpoint saves.')
  parser.add_argument('--save_summary_steps', type=int, default=100,
                      help='Number of steps between saving summaries.')
  parser.add_argument('--train_steps', type=int, default=100000,
                      help='Number of steps to train.')
  parser.add_argument('--use_tpu', type=bool, default=False,
                      help='Whether to train on TPU.')
  parser.add_argument('--shuffle_buffer_size', type=int, default=1024,
                      help='Number of images to read into buffer before '
                      'shuffling.')
  parser.add_argument('--tpu_master',
                      help='Location of TPU master.')
  args = parser.parse_args()

  main(args)
