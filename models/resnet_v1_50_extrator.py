# -*- coding: utf-8 -*-

from tensorflow.contrib.slim.nets import resnet_v1
import tensorflow as tf
import tensorflow.contrib.slim as slim
from sys import argv

import os

'''
    tfrecords_path = param_dict['tfrecords_path']
    pre_trained_model = param_dict['pre_trained_model']
    batch_size = param_dict['batch_size']
'''


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features={
            'image/class/label': tf.FixedLenFeature([], tf.int64),
            'image/encoded': tf.FixedLenFeature([], tf.string),
            'image/name': tf.FixedLenFeature([], tf.string)
        })
    img = tf.image.decode_jpeg(features['image/encoded'], channels=3)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    img = tf.image.resize_images(img, (224, 224), method=0)
    label = tf.cast(features['image/class/label'], tf.int32)
    name = tf.cast(features['image/name'], tf.string)
    return img, label, name


def inputs(file_name, batch_size, num_epochs):
    filename_queue = tf.train.string_input_producer([file_name], num_epochs=num_epochs)
    img, label, name = read_and_decode(filename_queue)

    _imgs, _labels, _names = tf.train.shuffle_batch([img, label, name], batch_size=batch_size,
                                                    num_threads=1, capacity=1000+3*batch_size,
                                                    min_after_dequeue=1000)
    return _imgs, _labels, _names


if __name__ == '__main__':
    # parse param
    print("argv: {}".format(argv))
    variables = argv[1:]
    param_dict = {}
    for var in variables:
        param_dict[var.split('=')[0]] = var.split('=')[1]
    print('param_dict: {}'.format(param_dict))


    # Create graph
    pre_trained_model = param_dict['pre_trained_model']
    batch_size = int(param_dict['batch_size'])
    vocab_file = os.path.join(param_dict['tensorboard_path'], 'vocab.txt')

    height = 224
    width = 224
    channels = 3
    imgs, labels, names = inputs(os.path.join(param_dict['tfrecords_path'], '0.tfrecords'), batch_size=batch_size,
                                 num_epochs=1)

    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        logits, end_points = resnet_v1.resnet_v1_50(imgs, is_training=False)


    # Restore model
    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
        saver.restore(sess, pre_trained_model)
        features = sess.graph.get_tensor_by_name('resnet_v1_50/pool5:0')
        count = 0
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            vocab_f = open(vocab_file, 'w+')
            while not coord.should_stop():
                fs, ls, ns = sess.run([features, labels, names])
                line = str(ls[0]) + ' ' + ' '.join([str(x) for x in fs.squeeze().tolist()]) + '\n'
                vocab_f.write(line)
            vocab_f.close()
        except tf.errors.OutOfRangeError:
            print('count %d done' % count)
        finally:
            coord.request_stop()
        coord.join(threads)

