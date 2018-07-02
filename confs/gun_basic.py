# -*- coding: utf-8 -*-
import gevent.monkey
import multiprocessing
gevent.monkey.patch_all()

bind = '0.0.0.0:8080'
debug = True
loglevel = 'debug'

pidfile = '/Users/loapui/Projects/PyCharmProjects/shoes_recognition/gunicorn.pid'
logfile = '/Users/loapui/Projects/PyCharmProjects/shoes_recognition/gunicorn.log'

workers = 1
worker_class = 'gevent'
backlog = 2048
chdir = '/Users/loapui/Projects/PyCharmProjects/shoes_recognition/service'
proc_name = 'shoes_gunicorn'
raw_env = ["IMAGENET_MODEL_PATH=/Users/loapui/Projects/PyCharmProjects/shoes_recognition/pretrained_models", "FASHION_SHOES_MODEL_PATH=/Users/loapui/Projects/PyCharmProjects/shoes_recognition/pretrained_models/own_model.h5"]
