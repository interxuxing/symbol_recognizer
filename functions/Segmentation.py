#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2 as cv
import numpy as np
import copy

# 运算符号分割

def preProcess(img,thresh=0.1):
    # 按列求和的等价表示 避免使用for循环。
    ysum = np.sum(img, axis=0)
    # 按每个【有像素的列】的白色点的值设置Threshold 用于截断
    ycount = len(ysum.nonzero()[0])
    Threshold = np.sum(ysum, axis=0) / ycount * thresh
    # 找出所有小于Threshold的值 并置为0
    ysum[np.where(ysum < Threshold)] = 0
    return ysum


def IsHead(ysum, idx):
    # 边界判断,第一的情形
    if idx == 0:
        if ysum[idx] > 0:
            return True
        else:
            return False
    # 中间行，此列0 下列>0 认为是新行开头
    if ysum[idx - 1] == 0 and ysum[idx] > 0:
        return True


def IsTail(ysum, idx):

    # 边界判断 最后一列的情形：
    if idx == len(ysum) - 1:
        if ysum[idx] > 0:
            return True
    if ysum[idx] > 0 and ysum[idx + 1] == 0:
        return True

# 使用某列的值和下一列比较，如果从0变成非0开始扫描字符，记为Head。
# 如果从 非0 变成0 说明字符扫描结束，记为Tail。


def findChar(ysum):
    lineHead = []
    lineTail = []
    for i in range(len(ysum)):
        if IsHead(ysum, i) is True:
            head = i
            lineHead.append(head)
        if IsTail(ysum, i) is True:
            tail = i
            lineTail.append(tail)
    return lineHead, lineTail


def charCut(image, hist):
    """

    :param image: 二值化图像
    :param hist: 投影直方图
    :return:
    """
    imgList = []
    # 获取字符区间
    H, T = findChar(hist)
    widthList=[]
    # 分割区间

    for i in range(len(H)):
        rec = image[0:image.shape[0],H[i]-2:T[i]+2]
        num_pixes = np.sum(rec/255)
        if num_pixes<20:
            continue
        if rec.size > 0:
            # print i
            # cv.imshow('img',rec)
            # cv.waitKey(0)
            imgList.append(rec)
            widthList.append(rec.shape[1])

    return imgList,widthList

def coarseSeg(image,thresh=0.1):
    hist = preProcess(image,thresh)
    imgList,widthList = charCut(image, hist)

    return imgList,widthList

def isFused(box_a,box_b,thresh=0.1):
    a_range = range(box_a[0],box_a[0]+box_a[2])
    b_range = range(box_b[0],box_b[0]+box_b[2])
    intersect = np.intersect1d(a_range,b_range)
    overlap_ratio_a = intersect.shape[0] / len(a_range)
    overlap_ratio_b = intersect.shape[0] / len(b_range)
#    overlap_ratio = 2*(overlap_ratio_a*overlap_ratio_b)/(overlap_ratio_a+overlap_ratio_b)
    if overlap_ratio_a >= thresh or overlap_ratio_b > thresh:
        return True
    return False

def boxFuse(box_a,box_b):
    fused_box = np.zeros_like(box_a)
    # left_top_x
    fused_box[1]= np.min([box_a[1],box_b[1]])
    # left_top_y
    fused_box[0] = np.min([box_a[0],box_b[0]])
    # box_w
    fused_box[2] = np.max([box_a[0]+box_a[2],box_b[0]+box_b[2]])
    # box_h
    fused_box[3] = np.max([box_a[1]+box_a[3],box_b[1]+box_b[3]])
    return fused_box

def boxFilter(boxSet):
    process_record = np.zeros((len(boxSet),))
    newBoxSet = []
    for i in np.arange(len(boxSet)):

        if process_record[i] == 0:

            for j in np.arange(len(boxSet)):
                if i==j or process_record[j]>0:
                    continue
                if isFused(boxSet[i],boxSet[j]):
                    boxSet[i] = boxFuse(boxSet[i],boxSet[j])
                    process_record[j] = 1
            newBoxSet.append(boxSet[i])
            process_record[i] = 1
    return newBoxSet

def genBoxes(contours):
    boxSet = []
    for cnt in contours:
        boxSet.append(np.array(cv.boundingRect(cnt)).reshape(4,))

    return boxSet

def refineImgList(imgList):
    """
    该函数对已分割图片集进行提炼
    :param imgList: 初次分割的图片集



    问题描述：
            第一次分割中由于连笔的原因，初次分割会使得部分分割的图片包含多个字符

    解决方法：
            计算每个图片中的轮廓。当轮廓数大于2，即可通过轮廓坐标进行二次分割

    """
    newImgList = []

    for i in np.arange(len(imgList)):

        img = copy.deepcopy(imgList[i])
        contours, hierarchy = cv.findContours(img,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

        boxSet = genBoxes(contours)
        # shown_img = np.zeros((img.shape[0],img.shape[1],3),dtype='uint8')
        # for cnt in contours:
        #
        #     cv.drawContours(shown_img,cnt,-1,(0,0,255),3)
        #     cv.imshow('img'+str(i),shown_img)
        #     cv.waitKey(0)
        # print i
        if len(boxSet)>=2:
            boxSet = boxFilter(boxSet)
            boxSet = np.asarray(boxSet)
            idx = np.argsort(boxSet[:,0])
            boxSet = boxSet[idx]
            for j in np.arange(boxSet.shape[0]):
                bb = boxSet[j]

                region = imgList[i][bb[1]:bb[1]+bb[3],bb[0]:bb[0]+bb[2],]
                idx = np.where(region==255)
                if idx[0].shape[0]<20:
                    continue
                newImgList.append(region)
        else:
            num_pixels = np.sum(imgList[i]/255.)
            if num_pixels>=20:
                newImgList.append(imgList[i])
        # if len(contours)>=2:
        #     for cnt in contours:
        #         rec = np.array(cv.boundingRect(cnt)).reshape(4,)
        #         if rec[2]*rec[3]>50:
        #             region = imgList[i][rec[1]:rec[1]+rec[3],rec[0]:rec[0]+rec[2],]
        #             newImgList.append(region)
        # else:
        #     newImgList.append(imgList[i])

    return newImgList

def imgSeg(image):
    # 二值化
    ret, bin_img = cv.threshold(image, 70, 255, cv.THRESH_BINARY)
    bin_img = cv.bitwise_not(bin_img)

    # element = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    # # 膨胀
    #
    # bin_img = cv.dilate(bin_img,element,iterations=1)
    # bin_img = cv.erode(bin_img, element,iterations=1)
    # cv.imshow('img',bin_img)
    # cv.waitKey(0)
    # 初分割
    imgList,widthList = coarseSeg(bin_img,0.1)
    # 提炼分割
    imgList = refineImgList(imgList)

    return imgList

# img = cv.imread('./test10.jpg', cv.IMREAD_GRAYSCALE)
# imgList = imgSeg(img)
# cv.namedWindow('img')
# cv.imshow('img',img)
# cv.waitKey(0)
# for i in np.arange(len(imgList)):
#     # cv.imwrite('./seg/test10/coarse/'+str(i)+'.bmp',imgList[i])
#     cv.imwrite('./seg/test10/refine/'+str(i)+'.bmp',imgList[i])
#     # cv.imshow('img'+str(i),imgList[i])
#     # cv.waitKey(0)