#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import Recornizer
import cv2 as cv
import Segmentation
import copy
import skimage.transform


# 待识别图片预处理
def imgPreprocess(img):
    idx = np.where(img==255)
    min_x = np.min(idx[0])
    min_y = np.min(idx[1])
    max_x = np.max(idx[0])
    max_y = np.max(idx[1])
    temp_img = img[min_x:max_x,min_y:max_y]
    cropped_img = np.zeros((temp_img.shape[0]+10,temp_img.shape[1]+10))
    cropped_img[5:5+temp_img.shape[0],5:5+temp_img.shape[1]] = temp_img
    return cropped_img


def getRecornizer():
    return Recornizer.getRecornizer()

def label2char(labels):
    semanticStr = ''
    for i in np.arange(len(labels)):
        if labels[i]<10:
            semanticStr = semanticStr + str(labels[i])[1]
        elif labels[i]==10:
            semanticStr = semanticStr + '+'
        elif labels[i]==11:
            semanticStr = semanticStr + '-'
        elif labels[i]==12:
            semanticStr = semanticStr + '*'
        elif labels[i]==13:
            semanticStr = semanticStr + '/'
        elif labels[i]==14:
            semanticStr = semanticStr + '('
        elif labels[i]==15:
            semanticStr = semanticStr + ')'
        elif labels[i]==16:
            semanticStr = semanticStr + '='
    return semanticStr

def recornizing(image):
    recornizer = getRecornizer()
    imgList = Segmentation.imgSeg(image)
    res = []
    for i in np.arange(len(imgList)):
        charImg = copy.deepcopy(imgList[i])
        charImg = imgPreprocess(charImg)
        charImg = skimage.transform.resize(charImg,(28,28),preserve_range=True)
        charImg /= 255.
        charImg = np.reshape(charImg,(1,1,28,28)).astype('float32')
        res.append(recornizer(charImg))
        # cv.imshow('img'+str(i),imgList[i])
        # cv.waitKey(0)
    resStr = label2char(res)
    return resStr

img = cv.imread('test10.jpg',cv.IMREAD_GRAYSCALE)
print recornizing(img)