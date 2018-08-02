# -*- coding: utf-8 -*-
#import gevent.monkey
#import multiprocessing
#gevent.monkey.patch_all()

bind = 'localhost:50001'
debug = False
loglevel = 'error'

access_log_format='%(t)s %(p)s %(h)s "%(r)s" %(s)s %(L)s %(b)s %(f)s" "%(a)s'
pidfile = '/Users/loapui/Projects/PyCharmProjects/shoes_recognition/logs/gunicorn.pid'
#logfile = '/usr/local/services/tensorflow_110-1.0/source/shoes_recognition/logs/gunicorn.log'
errorlog = '/Users/loapui/Projects/PyCharmProjects/shoes_recognition/logs/gunicorn_error.log'
accesslog = '/Users/loapui/Projects/PyCharmProjects/shoes_recognition/logs/gunicorn_access.log'


workers = 1
worker_class = 'gevent'
backlog = 15
chdir = '/Users/loapui/Projects/PyCharmProjects/shoes_recognition/service'
proc_name = 'shoes_gunicorn'
raw_env = ["IMAGENET_MODEL_PATH=/Users/loapui/Projects/PyCharmProjects/shoes_recognition/pretrained_models/own_model.h5", "FASHION_SHOES_MODEL_PATH=/Users/loapui/Projects/PyCharmProjects/shoes_recognition/pretrained_models/own_model.h5",
"FLASK_LOG_PATH=/Users/loapui/Projects/PyCharmProjects/shoes_recognition/logs", "get_url_timeout=500"]
