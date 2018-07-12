# -*- coding: utf-8 -*-

import argparse
import logging
import numpy as np
import os
import tensorflow as tf

from keras import applications
from gevent import monkey
from gevent import wsgi
from io import BytesIO
from requests import get
from flask import Flask, request as req
from flask_cors import CORS
from json import dumps
from keras.preprocessing.image import load_img, img_to_array
from keras.utils.generic_utils import CustomObjectScope
from keras.models import load_model, Model
from keras.preprocessing import image


app = Flask(__name__)
cors = CORS(app)
monkey.patch_all()
global model
global feature_model
global feature_datagen
global graph
# global logits
# global inputs
fashion_shoes_model_path = 'FASHION_SHOES_MODEL_PATH'
proxies = {
    "http": "http://10.139.11.75:80",
    "https": "http://10.139.11.75:80"
    # "http": "http://dev-proxy.oa.com:8080",
    # "https": "http://dev-proxy.oa.com:8080"
}

# RETURN
RET = {
    'GetUrlFail': -1,
    'GetFileFail': -2,
    'NotExpectInputType': -3,
    'IllegalApi': -4,
    'Crash': -5,
    'OK': 0
}


# logging.basicConfig(level=logging.ERROR)
handler = logging.FileHandler('../logs/flask.log', encoding='UTF-8')
handler.setLevel(logging.ERROR)
logging_format = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)s - %(message)s')
handler.setFormatter(logging_format)
app.logger.addHandler(handler)
# model_path = '/Users/loapui/models/own_model.h5'
model_path = ''
if fashion_shoes_model_path not in os.environ:
    app.logger.error('{} not in sys environ'.format(fashion_shoes_model_path))
    exit(1)
else:
    model_path = os.environ[fashion_shoes_model_path]

app.logger.error('load mobilenet with params from {}'.format(model_path))
with CustomObjectScope({'relu6': applications.mobilenet.relu6,
                        'DepthwiseConv2D': applications.mobilenet.DepthwiseConv2D}):
    model = load_model(model_path)

feature_model = Model(inputs=model.input, outputs=model.get_layer('logits').output)
feature_datagen = image.ImageDataGenerator(rescale=1./255)
graph = tf.get_default_graph()
app.logger.error('starting the api')

# logging.info('loading the model from {}'.format(model_path))
# x_placeholder = tf.placeholder(tf.float32, shape=[None, None, None, 3])
# x_resize_placeholder = tf.image.resize_images(x_placeholder, size=(224, 224))
# inputs = tf.map_fn(lambda x: tf.image.per_image_standardization(x), x_resize_placeholder)
# with slim.arg_scope(resnet_v1.resnet_arg_scope()):
#     _, _ = resnet_v1.resnet_v1_50(inputs, num_classes=1000)
# saver = tf.train.Saver()
# sess = tf.Session()
# # restore
# saver.restore(sess, model_path)
# feature = sess.graph.get_tensor_by_name('resnet_v1_50/pool5:0')


@app.errorhandler(403)
def page_not_found():
    return "403", "page_not_found"


@app.errorhandler(404)
def page_not_found(error):
    return "404", "page_not_found"


@app.errorhandler(400)
def page_not_found(error):
    return "400", "page_not_found"


@app.errorhandler(410)
def page_not_found(error):
    return "410", "page_not_found"


@app.errorhandler(500)
def page_not_found(error):
    return "500", "page_not_found"


@app.route("/fashion/shoes/features/<x>", methods=["POST", "GET"])
def extract(x):
    ret_code = RET['IllegalApi']
    feats = "illegal api"
    if req.method == 'GET':
        print(req.args.get('data'))
        if x == "url":
            ret_code, feats = predict_from_url(req.args.get('data'))
        elif x == "file":
            fp = BytesIO(req.args.get('data'))
            ret_code, feats = predict_from_file(fp)
    else:
        if x == "url":
            ret_code, feats = predict_from_url(req.data)
        elif x == "file":
            fp = BytesIO(req.data)
            ret_code, feats = predict_from_file(fp)

    response = {'ret': ret_code,
                'msg': feats}
    return dumps(response)
    # if success == 0:
    #     return dumps({"pred": feats})
    # else:
    #     return dumps({"error": feats})


def predict_from_url(url):
    app.logger.debug("download image file from %s", url)
    try:
        res = get(url, stream=True, timeout=200.0/1000)
        fp = BytesIO(res.content)
    except:
        app.logger.error("fail to get image from url {}".format(url))
        return RET['GetUrlFail'], "fail to get image from url"
    return predict_from_file(fp)


def predict_from_file(fp):
    try:
        # img = load_img(fp, target_size=(224, 224))
        img = load_img(fp, target_size=(224, 224))
        arr = img_to_array(img)
    except:
        app.logger.error("fail to load image from data")
        return RET['GetFileFail'], "fail to load image from data"
    return predict_from_arr(arr)


def predict_from_arr(arr):
    # if isinstance(arr, np.ndarray):
    #     if arr.shape == (224, 224, 3):
    #         x = np.expand_dims(arr, axis=0)
    #         # x = preprocess_input(x)
    #         feat = sess.run(feature, feed_dict={inputs: x})
    #         # y = np.argmax(pred_logits)
    #         y = list([float(x) for x in feat.squeeze()])
    #         return True, y
    #     else:
    #         return False, "expect an array with shape (224, 224, 3)"
    try:
        if isinstance(arr, np.ndarray):
            x = np.expand_dims(arr, axis=0)
            # feat = sess.run(feature, feed_dict={x_placeholder: x})
            with graph.as_default():
                for run in range(1):
                    for batch_x, _ in feature_datagen.flow(x, [0], batch_size=1):
                        feat = feature_model.predict(batch_x)
                        y = list([float(x) for x in feat.squeeze()])
                        break
            return RET['OK'], y
        app.logger.error("expect a numpy.ndarray")
        return RET['NotExpectInputType'], "expect a numpy.ndarray"
    except:
        app.logger.error("crash when predicting")
        return RET['Crash'], "crash when predicting"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bind-to", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=50001)
    # parser.add_argument("--model_path", type=str)
    args = parser.parse_args()

    app.logger.error('starting the api')
    # app.run(host=args.bind_to, port=args.port)
    server = wsgi.WSGIServer(('localhost', 50001), app)
    server.serve_forever()
