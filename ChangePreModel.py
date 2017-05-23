import os

import numpy as np
import tensorflow as tf
import c3d_model

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


class Par:
    def __init__(self, a=101, b=3):
        c3d_model.NUM_CLASSES = a
        c3d_model.CHANNELS = b

    def getW(self):
        with tf.variable_scope('var_name') as scope:
            # scope.reuse_variables()
            weights = {
                'wc1': train._variable_with_weight_decay('wc1', [3, 3, 3, c3d_model.CHANNELS, 64], 0.0005),
                'wc2': train._variable_with_weight_decay('wc2', [3, 3, 3, 64, 128], 0.0005),
                'wc3a': train._variable_with_weight_decay('wc3a', [3, 3, 3, 128, 256], 0.0005),
                'wc3b': train._variable_with_weight_decay('wc3b', [3, 3, 3, 256, 256], 0.0005),
                'wc4a': train._variable_with_weight_decay('wc4a', [3, 3, 3, 256, 512], 0.0005),
                'wc4b': train._variable_with_weight_decay('wc4b', [3, 3, 3, 512, 512], 0.0005),
                'wc5a': train._variable_with_weight_decay('wc5a', [3, 3, 3, 512, 512], 0.0005),
                'wc5b': train._variable_with_weight_decay('wc5b', [3, 3, 3, 512, 512], 0.0005),
                'wd1': train._variable_with_weight_decay('wd1', [8192, 4096], 0.0005),
                'wd2': train._variable_with_weight_decay('wd2', [4096, 4096], 0.0005),
                'wout': train._variable_with_weight_decay('wout', [4096, c3d_model.NUM_CLASSES], 0.0005)
            }
            biases = {
                'bc1': train._variable_with_weight_decay('bc1', [64], 0.000),
                'bc2': train._variable_with_weight_decay('bc2', [128], 0.000),
                'bc3a': train._variable_with_weight_decay('bc3a', [256], 0.000),
                'bc3b': train._variable_with_weight_decay('bc3b', [256], 0.000),
                'bc4a': train._variable_with_weight_decay('bc4a', [512], 0.000),
                'bc4b': train._variable_with_weight_decay('bc4b', [512], 0.000),
                'bc5a': train._variable_with_weight_decay('bc5a', [512], 0.000),
                'bc5b': train._variable_with_weight_decay('bc5b', [512], 0.000),
                'bd1': train._variable_with_weight_decay('bd1', [4096], 0.000),
                'bd2': train._variable_with_weight_decay('bd2', [4096], 0.000),
                'bout': train._variable_with_weight_decay('bout', [c3d_model.NUM_CLASSES], 0.000),
            }
        return weights, biases

# the first way to read model from file
def readMode():
    par1 = Par(249, 2)
    weights, biases = par1.getW()
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
        saver = tf.train.Saver(weights.values() + biases.values())
        init = tf.global_variables_initializer()
        sess.run(init)
        modelfilename = "./models/c3d_finetuning_new.model"
        modelfilename = "./models/c3d_ucf_model-17800"
        if os.path.isfile(modelfilename):
            saver.restore(sess, modelfilename)

        for name in weights.keys():
            print name + " " + weights[name].name + " " + str(weights[name].get_shape())
            # print np.shape(var)

        for name in biases.keys():
            print name + " " + biases[name].name + " " + str(biases[name].get_shape())


# the second way to read model from file
def readMode2():
    reader = tf.train.NewCheckpointReader("./models/sports1m_finetuning_ucf101.model")

    # get the value by name
    wc1 = reader.get_tensor("var_name/wc1")
    # print wc1

    # get the name of all variable in the model file
    # print reader.debug_string()
    variables = reader.get_variable_to_shape_map()
    for ele in variables:
        print ele

def compareTwoModel():
    modelfilename = "./models/sports1m_finetuning_ucf101.model"
    reader = tf.train.NewCheckpointReader("./models/c3d_ucf_model-17800.data-00000-of-00001")

    modelfilename = "./models/c3d_finetuning_new.model"
    reader2 = tf.train.NewCheckpointReader(modelfilename)

    variables = reader.get_variable_to_shape_map()
    for name in variables :
        print name
        var1 = reader.get_tensor(name)
        var2 = reader2.get_tensor(name)
        if np.shape(var1) != np.shape(var2):
            print "shape is not the same"
            continue
        if (var1 == var2).all():
            print "equal"
        else:
            print "no equal"

# change the pre-train model to that can be read by the code
# for those shape is not the same, init it by random
def changeModel():
    par = Par(249, 2)
    weights, biases = par.getW()
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
        # saver = tf.train.Saver(weights.values() + biases.values())
        init = tf.global_variables_initializer()
        sess.run(init)
        reader = tf.train.NewCheckpointReader("./models/sports1m_finetuning_ucf101.model")
        print reader.debug_string()

        data_dict = {}
        updates = []
        for name in weights.keys():
            print name
            var = reader.get_tensor("var_name/" + name)
            # print weights[name].get_shape()
            # print np.shape(var)
            data_dict[name] = var
            if weights[name].get_shape() == np.shape(var):
                updates.append(tf.assign(weights[name], var))
            else:
                data_dict[name] = sess.run(weights[name])
                print "shape is not the same"

        for name in biases.keys():
            print name
            var = reader.get_tensor("var_name/" + name)
            data_dict[name] = var
            if biases[name].get_shape() == np.shape(var):
                updates.append(tf.assign(biases[name], var))
            else:
                data_dict[name] = sess.run(biases[name])
                print "shape is not the same"

        for update in updates:
            sess.run(update)
        np.save("./models/c3d_finetuning_new2", data_dict)
        # saver.save(sess, "./models/c3d_finetuning_new.model")


if __name__ == '__main__':
    changeModel()
    # readMode()
    # readMode2()
    # compareTwoModel()

