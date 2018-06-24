# -*- coding: utf-8 -*-

import argparse
import logging
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import os


from keras.preprocessing.image import load_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from tensorflow.contrib.slim.nets import resnet_v1


def subtract_mean(image):
    _R_MEAN = 123.68
    _G_MEAN = 116.78
    _B_MEAN = 103.94
    ret = np.zeros(shape=image.shape)
    ret[:, :, 0] = image[:, :, 0] - _R_MEAN
    ret[:, :, 1] = image[:, :, 1] - _G_MEAN
    ret[:, :, 2] = image[:, :, 2] - _B_MEAN
    return ret


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


def get_batch(files, batch_size=32):
    num_files = len(files)
    n_epoches = num_files // batch_size
    for run in range(n_epoches+1):
        images = []
        if run == n_epoches:
            # last run
            batch_files = files[run*batch_size:]
            for pic in batch_files:
                img = load_img(pic, target_size=(224, 224))
                images.append(subtract_mean(img_to_array(img)))
            images = np.array(images)
            yield images, batch_files
        else:
            batch_files = files[run*batch_size:(run+1)*batch_size]
            for pic in batch_files:
                img = load_img(pic, target_size=(224, 224))
                images.append(subtract_mean(img_to_array(img)))
            images = np.array(images)
            yield images, batch_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--train_val_path", type=str, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    print('loading the model from {}'.format(args.model_path))
    inputs = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        logits, end_points = resnet_v1.resnet_v1_50(inputs, num_classes=1000)
    saver = tf.train.Saver()
    sess = tf.Session()
    # restore
    saver.restore(sess, args.model_path)

    print('loading train & val files from {}'.format(args.train_val_path))
    all_files = []
    with open(os.path.join(args.train_val_path, 'train.txt'), 'r') as f:
        for line in f.readlines():
            _file = line.strip().split(' ')[0]
            all_files.append(os.path.join(args.data_path, _file))
    with open(os.path.join(args.train_val_path, 'valid.txt'), 'r') as f:
        for line in f.readlines():
            _file = line.strip().split(' ')[0]
            all_files.append(os.path.join(args.data_path, _file))

    print('predict by resnet_v1_50')
    with open(os.path.join(args.predict_path, 'predict_results.txt'), 'w+') as f:
        count = 0
        run = 0
        for batch_images, batch_names in get_batch(all_files, batch_size=32):
            print('run: {}'.format(run))
            run += 1
            lines = []
            preds = sess.run(logits, feed_dict={inputs: batch_images})
            pred_topN = topN(preds, 5)
            for n, p in zip(batch_names, pred_topN):
                if (770 not in p) and (514 not in p):
                    line = n + '\n'
                    lines.append(line)
            count += len(lines)
            f.writelines(lines)
        print('in {} images find {} image are not shoes'.format(len(all_files), count))










