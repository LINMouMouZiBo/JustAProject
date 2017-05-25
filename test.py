import sys
import os  
sys.path = sys.path + ['/usr/local/anaconda/lib/python27.zip', '/usr/local/anaconda/lib/python2.7', '/usr/local/anaconda/lib/python2.7/plat-linux2', '/usr/local/anaconda/lib/python2.7/lib-tk', '/usr/local/anaconda/lib/python2.7/lib-old', '/usr/local/anaconda/lib/python2.7/lib-dynload', '/usr/local/anaconda/lib/python2.7/site-packages', '/usr/local/anaconda/lib/python2.7/site-packages/Sphinx-1.5.1-py2.7.egg', '/usr/local/anaconda/lib/python2.7/site-packages/setuptools-27.2.0-py2.7.egg']

# to choose which GPU to use, for one GPU,like "0", for more, separate index by ,
os.environ["CUDA_VISIBLE_DEVICES"]="2,3"
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

model_save_dir = './models'
var_dict = {}
use_pretrained_model = True
pretrained_file_name = "./models/c3d_finetuning_2Channels.npy"

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

def _variable_on_cpu(name, shape, initializer):
  with tf.device('/cpu:0'):
    if use_pretrained_model :
      var = tf.get_variable(name, initializer=data_dict[name])
    else:
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

  if use_pretrained_model:
      global data_dict
      data_dict = np.load(pretrained_file_name, encoding='latin1').item()

  with tf.Graph().as_default():

    images_placeholder, labels_placeholder = placeholder_inputs(
                    FLAGS.batch_size * gpu_num
                    )
    logits = []

    for gpu_index in range(0, gpu_num):
      with tf.device('/gpu:%d' % gpu_index):
        with tf.name_scope('%s_%d' % ('dextro-research', gpu_index)) as scope:
          with tf.variable_scope('var_name') as var_scope:
            weights = {
              'wc1': _variable_with_weight_decay('wc1', [3, 3, 3, c3d_model.CHANNELS, 64], 0.0005),
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

          logit = c3d_model.inference_c3d(
                          images_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size,:,:,:,:],
                          0.5,
                          FLAGS.batch_size,
                          weights,
                          biases
                          )

          logits.append(logit)
          tf.get_variable_scope().reuse_variables()
    logits = tf.concat(logits, 0)
    
    prediction = tf.argmax(logits, 1)

    with tf.variable_scope(tf.get_variable_scope(), reuse=False):

      init = tf.global_variables_initializer()

      # Create a session for running Ops on the Graph.
      sess = tf.Session(
                      config=tf.ConfigProto(
                                      allow_soft_placement=True,
                                      log_device_placement=False
                                      )
                      )
      sess.run(init)

      for i in range(vd.length):
        print i
        buf = []
        try:
            samples, video_name = vd.select_with_sampling(i)
        except Exception, e:
            print e.message
            continue
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
