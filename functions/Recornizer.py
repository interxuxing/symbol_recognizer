#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cPickle
import theano
import theano.tensor as T
# import CFG
from lasagne.layers import get_output
# import numpy as np
# import Models
# from lasagne.layers import set_all_param_values

# def load_guozhao():
#     input_var = theano.tensor.tensor4('input',theano.config.floatX)
#     net = Models.build_model(input_var)
#     params= np.load(CFG.MODEL_DIR)['best_p']
#     set_all_param_values(net,params)
#     prob = get_output(net,deterministic=True)
#     pred = T.argmax(prob,axis=1)
#     pred_fn = theano.function(inputs=[input_var],outputs=pred)
#     return pred_fn

# 全局设置： 模型路径，模型类别
MODEL_DIR = 'lenet_11_cropped.pkl'
MODEL_TYPE = 1

# 载入网络模型
def loadNet():

    with open(MODEL_DIR,'r') as f:
        net = cPickle.load(f)
    f.close()
    # net = Models.build_model()
    # params= np.load(CFG.MODEL_DIR)['best_p']
    # set_all_param_values(net,params)
    return net

# 构建LeNet识别器
def loadLeNetRecornizer(net):
    prob = get_output(net['prob'],deterministic=True)
    pred = T.argmax(prob,axis=1)
    pred_fn = theano.function(inputs=[net['Input'].input_var],outputs=pred)
    return pred_fn

# 获取识别器
def getRecornizer():
    if MODEL_TYPE==1:
        net = loadNet()
        recornizer = loadLeNetRecornizer(net)
        # recornizer = load_guozhao()
        return recornizer
    else:
        # not implemented....
        pass

