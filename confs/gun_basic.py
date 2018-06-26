bind = '0.0.0.0:8080'
workers = 20
backlog = 2048
worker_class = 'gevent'
debug = True
chdir = '/Users/loapui/Projects/PyCharmProjects/shoes_recognition/service'
proc_name = 'gunicorn_imagenet_predict.proc'
raw_env=["IMAGENET_MODEL_PATH=/Users/loapui/Projects/PyCharmProjects/shoes_recognition/pretrained_models/resnet_v1_50.ckpt","FASHION_SHOES_MODEL_PATH=/Users/loapui/Projects/PyCharmProjects/shoes_recognition/pretrained_models/model.ckpt-4999"]
