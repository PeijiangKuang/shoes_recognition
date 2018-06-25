# -*- coding: utf-8 -*-


def print_dict(_dict):
    print('=' * 20)
    for k in _dict:
        print('{} : {}'.format(k, _dict[k]))
    print('=' * 20)


def parse_argv(argv):
    print("argv: {}".format(argv))
    variables = argv[1:]
    param_dict = {}
    for var in variables:
        param_dict[var.split('=')[0]] = var.split('=')[1]
    print_dict(param_dict)
    return param_dict


def verify_params(params, param_dict):
    for x in params:
        if x not in param_dict:
            print('{} is not in params'.format(x))
            return 1


def load_imagenet_label_map():
    pass


def topN(scores, n=3, map_dict=None):
    scores = scores.squeeze()
    ret = []
    sorted_list = sorted(zip(scores, range(len(scores))), key=lambda t: t[0], reverse=True)[0:n]
    top_index = [x[1] for x in sorted_list]
    if map_dict:
        ret.append([map_dict[x] for x in top_index])
    else:
        ret.append(top_index)
    return ret