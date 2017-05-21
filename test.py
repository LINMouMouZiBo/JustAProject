import sys
import os  
sys.path = sys.path + ['/usr/local/anaconda/lib/python27.zip', '/usr/local/anaconda/lib/python2.7', '/usr/local/anaconda/lib/python2.7/plat-linux2', '/usr/local/anaconda/lib/python2.7/lib-tk', '/usr/local/anaconda/lib/python2.7/lib-old', '/usr/local/anaconda/lib/python2.7/lib-dynload', '/usr/local/anaconda/lib/python2.7/site-packages', '/usr/local/anaconda/lib/python2.7/site-packages/Sphinx-1.5.1-py2.7.egg', '/usr/local/anaconda/lib/python2.7/site-packages/setuptools-27.2.0-py2.7.egg']

# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Trains and Evaluates the MNIST network using a feed dictionary."""
# pylint: disable=missing-docstring
import os
import time
import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import c3d_model
import math
import numpy as np
import read_valid_data

# Basic model parameters as external flags.
flags = tf.app.flags
gpu_num = 1
#flags.DEFINE_float('learning_rate', 0.0, 'Initial learning rate.')
flags.DEFINE_integer('batch_size', 1, 'Batch size.')
FLAGS = flags.FLAGS
MOVING_AVERAGE_DECAY = 0.9999
model_save_dir = './models'

def placeholder_inputs(batch_size):
  """Generate placeholder variables to represent the input tensors.

  These placeholders are used as inputs by the rest of the model building
  code and will be fed from the downloaded data in the .run() loop, below.

  Args:
    batch_size: The batch size will be baked into both placeholders.

  Returns:
    images_placeholder: Images placeholder.
    labels_placeholder: Labels placeholder.
  """
  # Note that the shapes of the placeholders match the shapes of the full
  # image and label tensors, except the first dimension is now batch_size
  # rather than the full size of the train or test data sets.
  images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                         c3d_model.NUM_FRAMES_PER_CLIP,
                                                         c3d_model.CROP_SIZE,
                                                         c3d_model.CROP_SIZE,
                                                         c3d_model.CHANNELS))
  labels_placeholder = tf.placeholder(tf.int64, shape=(batch_size))
  return images_placeholder, labels_placeholder

def average_gradients(tower_grads):
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    grads = []
    for g, _ in grad_and_vars:
      expanded_g = tf.expand_dims(g, 0)
      grads.append(expanded_g)
    grad = tf.concat(grads, 0)
    grad = tf.reduce_mean(grad, 0)
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads

def tower_loss(name_scope, logit, labels):
  cross_entropy_mean = tf.reduce_mean(
                  tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logit)
                  )
  tf.summary.scalar(
                  name_scope + 'cross entropy',
                  cross_entropy_mean
                  )
  weight_decay_loss = tf.add_n(tf.get_collection('losses', name_scope))
  tf.summary.scalar(name_scope + 'weight decay loss', weight_decay_loss)
  tf.add_to_collection('losses', cross_entropy_mean)
  losses = tf.get_collection('losses', name_scope)

  # Calculate the total loss for the current tower.
  total_loss = tf.add_n(losses, name='total_loss')
  tf.summary.scalar(name_scope + 'total loss', total_loss)

  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.99, name='loss')
  with tf.variable_scope(tf.get_variable_scope(), reuse=False):
    loss_averages_op = loss_averages.apply(losses + [total_loss])
  with tf.control_dependencies([loss_averages_op]):
    total_loss = tf.identity(total_loss)
  return total_loss

def tower_acc(logit, labels):
  correct_pred = tf.equal(tf.argmax(logit, 1), labels)
  accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
  return accuracy

def _variable_on_cpu(name, shape, initializer):
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var

def _variable_with_weight_decay(name, shape, wd):
  var = _variable_on_cpu(name, shape, tf.contrib.layers.xavier_initializer())
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def run_testing():
  vd = read_valid_data.ValidationData(height = 112, width = 112)
  valid_file_handle = open('valid_result.txt', "w")
  # Get the sets of images and labels for training, validation, and
  # Tell TensorFlow that the model will be built into the default Graph.

  # Create model directory
  use_pretrained_model = True
  pretrained_file_name = 'c3d_ucf_model-17800'                                                                            
  with tf.Graph().as_default():
    global_step = tf.get_variable(
                    'global_step',
                    [],
                    initializer=tf.constant_initializer(0),
                    trainable=False
                    )
    images_placeholder, labels_placeholder = placeholder_inputs(
                    FLAGS.batch_size * gpu_num
                    )
    tower_grads1 = []
    tower_grads2 = []
    logits = []
    opt1 = tf.train.AdamOptimizer(1e-5)
    opt2 = tf.train.AdamOptimizer(2e-5)
    for gpu_index in range(0, gpu_num):
      with tf.device('/gpu:%d' % gpu_index):
        with tf.name_scope('%s_%d' % ('dextro-research', gpu_index)) as scope:
          with tf.variable_scope('var_name') as var_scope:
            weights = {
              'wc1': _variable_with_weight_decay('wc1', [3, 3, 3, 2, 64], 0.0005),
              'wc2': _variable_with_weight_decay('wc2', [3, 3, 3, 64, 128], 0.0005),
              'wc3a': _variable_with_weight_decay('wc3a', [3, 3, 3, 128, 256], 0.0005),
              'wc3b': _variable_with_weight_decay('wc3b', [3, 3, 3, 256, 256], 0.0005),
              'wc4a': _variable_with_weight_decay('wc4a', [3, 3, 3, 256, 512], 0.0005),
              'wc4b': _variable_with_weight_decay('wc4b', [3, 3, 3, 512, 512], 0.0005),
              'wc5a': _variable_with_weight_decay('wc5a', [3, 3, 3, 512, 512], 0.0005),
              'wc5b': _variable_with_weight_decay('wc5b', [3, 3, 3, 512, 512], 0.0005),
              'wd1': _variable_with_weight_decay('wd1', [8192, 4096], 0.0005),
              'wd2': _variable_with_weight_decay('wd2', [4096, 4096], 0.0005),
              'out': _variable_with_weight_decay('wout', [4096, c3d_model.NUM_CLASSES], 0.0005)
              }
            biases = {
              'bc1': _variable_with_weight_decay('bc1', [64], 0.000),
              'bc2': _variable_with_weight_decay('bc2', [128], 0.000),
              'bc3a': _variable_with_weight_decay('bc3a', [256], 0.000),
              'bc3b': _variable_with_weight_decay('bc3b', [256], 0.000),
              'bc4a': _variable_with_weight_decay('bc4a', [512], 0.000),
              'bc4b': _variable_with_weight_decay('bc4b', [512], 0.000),
              'bc5a': _variable_with_weight_decay('bc5a', [512], 0.000),
              'bc5b': _variable_with_weight_decay('bc5b', [512], 0.000),
              'bd1': _variable_with_weight_decay('bd1', [4096], 0.000),
              'bd2': _variable_with_weight_decay('bd2', [4096], 0.000),
              'out': _variable_with_weight_decay('bout', [c3d_model.NUM_CLASSES], 0.000),
              }
          varlist1 = weights.values()
          varlist2 = biases.values()
          logit = c3d_model.inference_c3d(
                          images_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size,:,:,:,:],
                          0.5,
                          FLAGS.batch_size,
                          weights,
                          biases
                          )
          loss = tower_loss(
                          scope,
                          logit,
                          labels_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size]
                          )
          grads1 = opt1.compute_gradients(loss, varlist1)
          grads2 = opt2.compute_gradients(loss, varlist2)
          tower_grads1.append(grads1)
          tower_grads2.append(grads2)
          logits.append(logit)
          tf.get_variable_scope().reuse_variables()
    logits = tf.concat(logits, 0)
    
    prediction = tf.argmax(logits, 1)
    accuracy = tower_acc(logits, labels_placeholder)
    tf.summary.scalar('accuracy', accuracy)
    grads1 = average_gradients(tower_grads1)
    grads2 = average_gradients(tower_grads2)
    print grads1
    with tf.variable_scope(tf.get_variable_scope(), reuse=False):
      apply_gradient_op1 = opt1.apply_gradients(grads1)
      apply_gradient_op2 = opt2.apply_gradients(grads2, global_step=global_step)
      variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
      variables_averages_op = variable_averages.apply(tf.trainable_variables())
      train_op = tf.group(apply_gradient_op1, apply_gradient_op2, variables_averages_op)
      null_op = tf.no_op()

      # Create a saver for writing training checkpoints.
      saver = tf.train.Saver(weights.values() + biases.values())
      init = tf.global_variables_initializer()

      # Create a session for running Ops on the Graph.
      sess = tf.Session(
                      config=tf.ConfigProto(
                                      allow_soft_placement=True,
                                      log_device_placement=True
                                      )
                      )
      sess.run(init)
      if use_pretrained_model:
        saver.restore(sess, os.path.join(model_save_dir, pretrained_file_name))

      for i in range(vd.length):
        print i
        buf = []
        samples, video_name = vd.select_with_sampling(i)
        buf.append(video_name)
        for j in range(len(samples)):
          sample = samples[j]['sample']
          start = samples[j]['start']
          end = samples[j]['end']
          label = sess.run(prediction, feed_dict = {
              images_placeholder: [sample],
            })[0] + 1
          buf.append(str(start) + ',' + str(end) + ':' + str(label))

        valid_file_handle.write(' '.join(buf) + '\n')
    print("done")
    valid_file_handle.close()

def main(_):
  run_testing()

if __name__ == '__main__':
  tf.app.run()
