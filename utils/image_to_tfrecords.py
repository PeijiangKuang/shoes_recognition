# -*- coding: utf-8 -*-

import os
import tensorflow as tf
from sys import argv

'''
    image_path = param_dict['image_path']
    train_valid = param_dict['train_valid']
    tfrecords_path = param_dict['tfrecords_path']
'''


# to int64 feature
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# to bytes feature
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# png 2 jpeg
def png_to_jpeg(image_data):
    with tf.Session() as sess:
        image = tf.image.decode_png(image_data, channels=3)
        image = tf.image.encode_jpeg(image, format='rgb', quality=100)
    return sess.run(image)


# utils parse argv
def parse_argv(argv):
    print("argv: {}".format(argv))
    variables = argv[1:]
    param_dict = {}
    for var in variables:
        param_dict[var.split('=')[0]] = var.split('=')[1]
    print('param_dict: {}'.format(param_dict))
    return param_dict


# images to tfrecords
def convert_images_to_tfrecords(image_path, train_valid, tfrecords_path):
    count = 0
    writer = tf.python_io.TFRecordWriter(os.path.join(tfrecords_path, '0.tfrecords'))
    with open(os.path.join(train_valid, 'train.txt'), 'r') as _file:
        for line in _file:
            image_name, label = line.strip().split(' ')
            full_image_name = os.path.join(image_path, image_name)
            if not tf.gfile.Exists(full_image_name):
                print('image %s does not exists.' % full_image_name)
                continue
            with tf.gfile.FastGFile(full_image_name, 'rb') as f:
                img = f.read()
                if full_image_name.split('.')[1].lower() == 'png':
                    img = png_to_jpeg(img)
                label = int(label)
                example = tf.train.Example(features=tf.train.Features(feature={
                    'image/class/label': _int64_feature(label),
                    'image/encoded': _bytes_feature(img),
                    'image/name': _bytes_feature(image_name)
                }))
                writer.write(example.SerializeToString())
                count += 1
                if count % 500 == 0:
                    print('process %i images.' % count)

    with open(os.path.join(train_valid, 'valid.txt'), 'r') as _file:
        for line in _file:
            image_name, label = line.strip().split(' ')
            full_image_name = os.path.join(image_path, image_name)
            if not tf.gfile.Exists(full_image_name):
                print('image %s does not exists.' % full_image_name)
                continue
            with tf.gfile.FastGFile(full_image_name, 'rb') as f:
                img = f.read()
                if full_image_name.split('.')[1].lower() == 'png':
                    img = png_to_jpeg(img)
                label = int(label)
                example = tf.train.Example(features=tf.train.Features(feature={
                    'image/class/label': _int64_feature(label),
                    'image/encoded': _bytes_feature(img),
                    'image/name': _bytes_feature(image_name)
                }))
                writer.write(example.SerializeToString())
                count += 1
                if count % 500 == 0:
                    print('process %i images.' % count)
    
    writer.close()
    print('convert finished.')


if __name__ == '__main__':
    param_dict = parse_argv(argv)
    convert_images_to_tfrecords(image_path=param_dict['image_path'],
                                train_valid=param_dict['train_valid'],
                                tfrecords_path=param_dict['tfrecords_path'])
