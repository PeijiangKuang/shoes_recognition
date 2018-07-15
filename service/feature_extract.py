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
fashion_shoes_model_path = 'FASHION_SHOES_MODEL_PATH'

# RETURN
RET = {
    'GetUrlFail': -1,
    'GetFileFail': -2,
    'NotExpectInputType': -3,
    'IllegalApi': -4,
    'Crash': -5,
    'OK': 0
}


handler = logging.FileHandler(os.path.join(os.environ['FLASK_LOG_PATH'], 'flask.log'), encoding='UTF-8')
handler.setLevel(logging.ERROR)
logging_format = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s' +
                                   ' - %(funcName)s - %(lineno)s - %(message)s')
handler.setFormatter(logging_format)
app.logger.addHandler(handler)
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

del model

graph = tf.get_default_graph()
app.logger.error('starting the api')


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


def predict_from_url(url):
    try:
        res = get(url, stream=True, timeout=200.0/1000)
        fp = BytesIO(res.content)
    except:
        app.logger.error("fail to get image from url {}".format(url))
        return RET['GetUrlFail'], "fail to get image from url"
    return predict_from_file(fp)


def predict_from_file(fp):
    try:
        img = load_img(fp, target_size=(224, 224))
        del fp
        arr = img_to_array(img)
        del img
    except:
        app.logger.error("fail to load image from data")
        return RET['GetFileFail'], "fail to load image from data"
    return predict_from_arr(arr)


def predict_from_arr(arr):
    try:
        if isinstance(arr, np.ndarray):
            x = np.expand_dims(arr, axis=0)
            y = None
            feature_datagen = image.ImageDataGenerator(rescale=1. / 255)
            with graph.as_default():
                for run in range(1):
                    for batch_x, _ in feature_datagen.flow(x, [0], batch_size=1):
                        feat = feature_model.predict(batch_x)
                        y = list([float(x) for x in feat.squeeze()])
                        break

            del arr
            del feature_datagen

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
    args = parser.parse_args()

    app.logger.error('starting the api')
    server = wsgi.WSGIServer(('localhost', 50001), app)
    server.serve_forever()
