# -*- coding: utf-8 -*-

import os
from sys import argv

import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_v1
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import graph_util

'''
    ckpt_path = param_dict['model_path']
    pb_path = param_dict['pb_path']
'''

if __name__=='__main__':
    # parse param
    print("argv: {}".format(argv))
    variables = argv[1:]
    param_dict = {}
    for var in variables:
        param_dict[var.split('=')[0]] = var.split('=')[1]
    print('param_dict: {}'.format(param_dict))

    # verify params
    for x in ['pb_path', 'model_path']:
        if x not in param_dict:
            print('{} is not in params'.format(x))
            exit(1)

    # Create graph
    inputs = tf.placeholder(dtype=tf.float32, shape=(None, 224, 224, 3), name='inputs')
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        logits, end_points = resnet_v1.resnet_v1_50(inputs, is_training=False)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        ckpt = os.path.join(param_dict['model_path'], 'model.ckpt-4999')
        saver.restore(sess, ckpt)
        print('Succesfully loaded model from %s.' % ckpt)

        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['resnet_v1_50/pool5'])

        with tf.gfile.FastGFile(param_dict['pb_path'] + 'model.pb', mode='wb') as f:
            f.write(constant_graph.SerializeToString())

