# -*- coding: utf-8 -*-

import os
import argparse
import logging
import shutil
import imghdr
import numpy as np
import random

from PIL import Image
from keras.preprocessing import image

logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, required=True)
parser.add_argument("--recover", type=int, default=0)
parser.add_argument("--split", type=int, default=0)
parser.add_argument("--seg", type=int, default=None)
parser.add_argument("--filter", type=int, default=0)
parser.add_argument("--enhance", type=int, default=0)
parser.add_argument("--enhance_dirs", action='append', default=[])
parser.add_argument("--trainval", type=int, default=0)
parser.add_argument("--class_nums", type=int, default=1000)
parser.add_argument("--trainvalrate", type=float, default=0.2)
args = parser.parse_args()

if args.split == 1:
    if args.seg is not None and args.seg < 4:
        logging.info('='*10 + 'SPLIT' + '='*10)
        all_files = os.listdir(args.data)
        for _file in all_files:
            if not os.path.isfile(os.path.join(args.data, _file)):
                continue
            cate = _file.split('_')[args.seg]
            if not os.path.exists(os.path.join(args.data, cate)):
                os.makedirs(os.path.join(args.data, cate))
            shutil.move(os.path.join(args.data, _file), os.path.join(os.path.join(args.data, cate), _file))
    else:
        logging.error('when split is 1, seg is required and seg must less than 4')
        exit(1)

elif args.recover == 1:
    logging.info('='*10 + 'RECOVER' + '='*10)
    all_dirs = os.listdir(args.data)
    for _dir in all_dirs:
        all_files = os.listdir(os.path.join(args.data, _dir))
        for _file in all_files:
            shutil.move(os.path.join(os.path.join(args.data, _dir), _file), os.path.join(args.data, _file))

elif args.filter == 1:
    logging.info('='*10 + 'FILTER' + '='*10)
    if not os.path.exists(os.path.join(args.data, 'abandon')):
        os.mkdir(os.path.join(args.data, 'abandon'))
    all_files = os.listdir(args.data)
    for _file in all_files:
        if not os.path.isfile(_file):
            continue
        if imghdr.what(os.path.join(args.data, _file)) not in ['jpeg', 'png']:
            shutil.move(os.path.join(args.data, _file), os.path.join(os.path.join(args.data, 'abandon'), _file))

elif args.enhance == 1:
    if len(args.enhance_list) > 0:
        logging.info('='*10 + 'ENHANCE' + '='*10)
        for _dir in args.enhance_list:
            id_2_max = {}
            all_files = os.listdir(os.path.join(args.data, _dir))
            for _file in all_files:
                id = _file.split('_')[0] + '_' + _file.split('_')[1] + '_' + _file.split('_')[2]
                index = int(_file.split('.')[0].split('_')[3])
                if id not in id_2_max:
                    id_2_max[id] = index
                elif id_2_max[id] < index:
                    id_2_max[id] = index

            rotation_datagen = image.ImageDataGenerator(rotation_range=30)
            shear_datagen = image.ImageDataGenerator(shear_range=0.5)
            for _file in all_files:
                img = image.load_img(os.path.join(os.path.join(args.data, _dir), _file))
                id = _file.split('_')[0] + '_' + _file.split('_')[1] + '_' + _file.split('_')[2]
                max_index = id_2_max[id]
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                for run in range(1):
                    for batch_x, _ in rotation_datagen.flow(x, [0], batch_size=1):
                        batch_x = batch_x[0]
                        img = Image.fromarray(batch_x)
                        img.save(os.path.join(os.path.join(args.data, _dir), id+'_'+str(max_index+1)+'.jpg'))
                        break
                for run in range(1):
                    for batch_x, _ in shear_datagen.flow(x, [0], batch_size=1):
                        batch_x = batch_x[0]
                        img = Image.fromarray(batch_x)
                        img.save(os.path.join(os.path.join(args.data, _dir, id+'_'+str(max_index+2)+'.jpg')))
                        break
                id_2_max[id] = max_index+2
    else:
        logging.error('when enhance is 1, enhance_list can not be empty')

elif args.trainval == 1:
    logging.info('='*10 + 'TRAINVAL' + '='*10)
    branch_2_count = {}
    all_files = os.listdir(args.data)
    for _file in all_files:
        branch = _file.split('_')[0]
        if branch not in branch_2_count:
            branch_2_count[branch] = 0
        branch_2_count[branch] += 1

    trainval_branch_2_count = {}
    for k in branch_2_count:
        trainval_branch_2_count[k] = branch_2_count[k] * args.trainvalrate + args.class_nums

    if not os.path.exists(os.path.join(args.data, 'train')):
        os.mkdir(os.path.join(args.data, 'train'))
    if not os.path.exists(os.path.join(args.data, 'val')):
        os.mkdir(os.path.join(args.data, 'val'))

    for k in branch_2_count:
        logging.info('  processing {}'.format(k))
        if not os.path.exists(os.path.join(os.path.join(args.data, 'train'), k)):
            os.mkdir(os.path.join(os.path.join(args.data, 'train'), k))
        if not os.path.exists(os.path.join(os.path.join(args.data, 'val'), k)):
            os.mkdir(os.path.join(os.path.join(args.data, 'val'), k))

        branch_list = []
        for _file in all_files:
            branch = _file.split('_')[0]
            if branch == k:
                branch_list.append(_file)
        selected_branch_list = np.random.shuffle(branch_list)[:trainval_branch_2_count[k]]
        train_branch_list = selected_branch_list[:1000]
        val_branch_list = selected_branch_list[1000:]

        for _branch_file in train_branch_list:
            shutil.move(os.path.join(args.data, _branch_file),
                        os.path.join(os.path.join(os.path.join(args.data, 'train'), k), _branch_file))
        for _branch_file in val_branch_list:
            shutil.move(os.path.join(args.data, _branch_file),
                        os.path.join(os.path.join(os.path.join(args.data, 'val'), k), _branch_file))

else:
    logging.error('one of [split, recover] should be 1')
    exit(1)
