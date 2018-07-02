# Tensorflow Estimator example

This is a simple example of how to use Tensorflow's `Estimator` to train a model
on MNIST using the TFRecords data format.  There are two scripts here:
`convert_to_records.py`, which will get the MNIST data and convert it to
TFRecords format; and `trainer.py`, which will train on MNIST.

Example usage:

```
convert_to_records.py --data_dir=$HOME/data/mnist
```

```
trainer.py \
  --batch_size=128 \
  --train_file_pattern=$HOME/data/mnist/train.tfrecords \
  --eval_file_pattern=$HOME/data/mnist/test.tfrecords \
  --eval_freq=100 \
  --model_dir=/tmp/mnist \
  --train_steps=10000 \
  --save_summary_steps=10
  --shuffle_buffer_size=60000 \
  --learning_rate=0.0001 \
  --mode=train_and_eval
```
