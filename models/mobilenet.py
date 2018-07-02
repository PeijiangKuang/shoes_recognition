# -*- coding: utf-8 -*-

"""
    data/
        train/
            cate1/
                cate1_001.jpg
                cate1_002.jpg
                ...
            cate2/
                cate2_001.jpg
                cate2_002.jpg
                ...
        validation/
            cate1/
                cate1_001.jpg
                cate2_002.jpg
                ...
            cate2/
                cate2_001.jpg
                cate2_002.jpg
                ...
"""

import argparse
import os
import logging
import numpy as np
import keras

from keras.applications.mobilenet import MobileNet
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras import regularizers
from keras.utils.generic_utils import CustomObjectScope

# logging config
logging.basicConfig(level=logging.INFO)

# argument parser
parser = argparse.ArgumentParser()
# parser.add_argument("--data", type=str, required=True)
parser.add_argument("--train", type=str, default=None)
parser.add_argument("--feature", type=str, default=None)
parser.add_argument("--predict", type=str, default=None)
parser.add_argument("--from_model", type=str, default=None)
parser.add_argument("--tensorboard", type=str, default=None)
parser.add_argument("--num_classes", type=int, default=1000)
args = parser.parse_args()


if args.from_model:
    logging.info('load mobilenet with params from {}'.format(args.from_model))
    with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,
                            'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
        model = load_model(args.from_model)
else:
    home_path = '/cephfs/group/one-sng-qzone-online-feedranking/kennykuang/workspace/model'
    model = 'mobilenet_1_0_224_tf_no_top.h5'
    from_model = os.path.join(home_path, model)
    logging.info('load mobilenet with params from {}'.format(from_model))
    mobilenet_model = MobileNet(weights=from_model, include_top=False)
    logging.info('add own classfier')
    x = mobilenet_model.output
    x = GlobalAveragePooling2D()(x)
    # x = Dropout(0.5, name='dropout_1')(x)
    x = Dense(1024, activation='relu', name='logits', kernel_initializer='glorot_normal',
              kernel_regularizer=regularizers.l1(0.001))(x)
    x = Dropout(0.5, name='dropout_2')(x)
    predictions = Dense(args.num_classes, activation='softmax', name='predictions',
                        kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l1(0.001))(x)
    model = Model(inputs=mobilenet_model.input, outputs=predictions)

for i, layer in enumerate(model.layers):
    print('{}: {}({})'.format(i, layer.name, layer.output_shape))

if args.train is not None:
    logging.info('='*10 + 'TRAIN FROM PATH {}'.format(args.train) + '='*10)
    # logging.info('lock mobilenet convs layers')
    # for layer in model.layers[:64]:
    #     layer.trainable = False
    # for layer in model.layers[64:]:
    #     layer.trainable = True

    logging.info('compile mobilenet')
    adam_optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=adam_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    logging.info('train & valid data generator')
    train_datagen = image.ImageDataGenerator(rescale=1./255, horizontal_flip=True)
    valid_datagen = image.ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        os.path.join(args.train,  'train'),
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        seed=7777)
    validation_generator = valid_datagen.flow_from_directory(
        os.path.join(args.train, 'val'),
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        seed=7777)

    logging.info('start training')
    if args.tensorboard is None:
        model.fit_generator(train_generator, epochs=20, validation_data=validation_generator)
    else:
        model.fit_generator(train_generator, epochs=20, validation_data=validation_generator,
                            callbacks=[TensorBoard(log_dir=args.tensorboard)])

    logging.info('save model')
    model.save('/cephfs/group/one-sng-qzone-online-feedranking/kennykuang/workspace/model/own_model.h5')

elif args.predict is not None:
    logging.info('='*10 + 'PREDICT FROM PATH {}'.format(args.predict) + '='*10)
    pred_datagen = image.ImageDataGenerator(rescale=1./255)
    for _file in os.listdir(args.predict_path):
        if os.path.isfile(os.path.join(args.predict_path, _file)):
            img = image.load_img(os.path.join(args.predict_path, _file), target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            for run in range(1):
                for batch_x, _ in pred_datagen.flow(x, [0], batch_size=1):
                    y = model.predict(batch_x).squeeze()
                    logging.info('{} predicts {}'.format(os.path.join(args.predict_path, _file), y))
                    break

elif args.feature is not None:
    logging.info('='*10 + 'FEATURE FROM PATH {}'.format(args.feature) + '='*10)
    feature_model = Model(inputs=model.input, outputs=model.get_layer('logits').output)
    feature_datagen = image.ImageDataGenerator(rescale=1./255)
    with open(os.path.join(args.feature, 'features.txt'), 'w+') as f:
        file_index = 0
        for _file in os.listdir(args.feature):
            try:
                if os.path.isfile(os.path.join(args.feature, _file)):
                    file_index += 1
                    logging.info('{} -> extracting feature from {}'.format(file_index, os.path.join(args.feature, _file)))
                    img = image.load_img(os.path.join(args.feature, _file), target_size=(224, 224))
                    x = image.img_to_array(img)
                    x = np.expand_dims(x, axis=0)
                    for run in range(1):
                        for batch_x, _ in feature_datagen.flow(x, [0], batch_size=1):
                            feature = feature_model.predict(batch_x).squeeze()
                            # logging.info('{} feature shape {}'.format(os.path.join(args.feature, _file),
                            # feature.shape))
                            break
                    lines = []
                    for i, val in enumerate(feature):
                        line = str(i) + '\t' + _file + '\t' + str(val) + '\t' + '100000001' + '\n'
                        lines.append(line)
                    f.writelines(lines)
            except:
                logging.info('failed to extract feature from {}'.format(os.path.join(args.feature, _file)))
                continue

else:
    logging.error('one of [train, predict, feature] should not be None')
    exit(1)
