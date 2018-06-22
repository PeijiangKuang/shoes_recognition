# -*- coding: utf-8 -*-


import os
from sys import argv

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_v1
import tensorflow.contrib.slim as slim



'''
    model_path = param_dict['model_path']
    label_map = param_dict['label_map']
'''


def load_label_map(label_map):
    map_dict = {}
    with open(label_map, 'r') as f:
        for line in f.readlines():
            spt = line.strip().split(' ')
            map_dict[int(spt[1])] = spt[0]
    return map_dict


def topN(scores, n=3, map_dict=None):
    scores = scores.squeeze()
    ret = []
    for i in range(scores.shape[0]):
        sorted_list = sorted(zip(scores[i], range(scores.shape[1])), key=lambda t: t[0], reverse=True)[0:n]
        top_index = [x[1] for x in sorted_list]
        if map_dict:
            ret.append([map_dict[x] for x in top_index])
        else:
            ret.append(top_index)
    return ret


def subtract_mean(image):
    _R_MEAN = 123.68
    _G_MEAN = 116.78
    _B_MEAN = 103.94
    ret = np.zeros(shape=image.shape)
    ret[:, :, 0] = image[:, :, 0] - _R_MEAN
    ret[:, :, 1] = image[:, :, 1] - _G_MEAN
    ret[:, :, 2] = image[:, :, 2] - _B_MEAN
    return image


def get_batch(files, batch_size=32):
    num_files = len(files)
    n_epoches = num_files // batch_size
    for run in range(n_epoches+1):
        images = []
        if run == n_epoches:
            # last run
            batch_files = files[run*batch_size:]
            for pic in batch_files:
                img = Image.open(pic)
                images.append(subtract_mean(np.array(img.convert('RGB').resize((224, 224)))))
            images = np.array(images)
            yield images, batch_files
        else:
            batch_files = files[run*batch_size:(run+1)*batch_size]
            for pic in batch_files:
                img = Image.open(pic)
                images.append(subtract_mean(np.array(img.convert('RGB').resize((224, 224)))))
            images = np.array(images)
            yield images, batch_files


if __name__ == '__main__':

    # parse param
    print("argv: {}".format(argv))
    variables = argv[1:]
    param_dict = {}
    for var in variables:
        param_dict[var.split('=')[0]] = var.split('=')[1]
    print('param_dict: {}'.format(param_dict))

    # verify params
    for x in ['label_map', 'model_path', 'pic_path', 'feature_path']:
        if x not in param_dict:
            print('{} is not in params'.format(x))
            exit(1)

    # load label_map
    if not os.path.exists(param_dict['label_map']):
        print('{} is not exists'.format(param_dict['label_map']))
        exit(1)
    label_map = load_label_map(param_dict['label_map'])

    # graph
    inputs = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        logits, end_points = resnet_v1.resnet_v1_50(inputs, num_classes=1000, is_training=False)
    saver = tf.train.Saver()

    # session run
    with tf.Session() as sess:
        # restore
        ckpt = os.path.join(param_dict['model_path'], 'resnet_v1_50.ckpt')
        saver.restore(sess, ckpt)
        print('Succesfully loaded model from %s.' % ckpt)

        files = list(map(lambda x: os.path.join(param_dict['pic_path'], x), os.listdir(param_dict['pic_path'])))
        with open(os.path.join(param_dict['feature_path'], 'features.txt'), 'w+') as f:
            for batch_images, batch_file_names in get_batch(files, 32):
                feed_dict = {inputs: batch_images}
                l = sess.run(logits, feed_dict=feed_dict)
                pred_topN = topN(l, 3)
                for n, preds in zip(batch_file_names, pred_topN):
                    print('{} predicts {}'.format(n, preds))
                print("=" * 10)

                # feature_tensor = sess.graph.get_tensor_by_name('resnet_v1_50/pool5:0')
                # feature = sess.run(feature_tensor, feed_dict=feed_dict).squeeze()
                # for n, feat in zip(batch_file_names, feature):
                #     line = n + ' ' + ' '.join([str(x) for x in feat]) + '\n'
                #     f.write(line)




