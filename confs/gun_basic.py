# -*- coding: utf-8 -*-

import gevent.monkey
import multiprocessing
gevent.monkey.patch_all()


bind = '0.0.0.0:8080'
debug = True
loglevel = 'debug'

pidfile = '.../gunicorn.pid'
logfile = '.../gunicorn.log'

workers = multiprocessing.cpu_count() * 2 + 1
worker_class = 'gunicorn.workers.ggevent.GeventWorker'
backlog = 2048
chdir = '.../service'
proc_name = '...'
raw_env = ["IMAGENET_MODEL_PATH=.../resnet_v1_50.ckpt", "FASHION_SHOES_MODEL_PATH=.../model.ckpt-4999"]