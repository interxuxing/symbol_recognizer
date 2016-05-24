#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2 as cv
import numpy as np



class SegTools:

    def __init__(self,img):
        self.img = img
        self.bin_img = self.__rgb2bin__()
        self.hist = self.__getHit__()
        # hist segment thresh hold
        self.firstThresh = 0.
        #crest thresh hold
        self.secondThresh = 0.5
        #mean lenth pixs
        self.thirdThresh = 0.73
        #crest thresh
        self.fourThresh = 0.35
        # mean lenth of
        self.fiveThresh = 0.5
        # other crest
        self.sixThresh = 0.28

    def __rgb2bin__(self):
        kernel1 = cv.getStructuringElement(cv.MORPH_ELLIPSE,(7,7))
        close = cv.morphologyEx(self.img,cv.MORPH_CLOSE,kernel1)
        div = np.float32(self.img)/(close)
        res = np.uint8(cv.normalize(div,div,0,255,cv.NORM_MINMAX))
        ret, bin_img = cv.threshold(res, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
        bin_img = cv.bitwise_not(bin_img)
        kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
        bin_img = cv.morphologyEx(bin_img , cv.MORPH_CLOSE, kernel)
        self.showImg(bin_img,'sb')
        return bin_img

    def __getHit__(self):

         ysum = np.sum(self.bin_img, axis=0)
         ysum /=255.
         return ysum

    def __segByHist__(self):
       indx = np.where(self.hist>self.firstThresh)
       self.firstSeg = []
       a = indx[0][0]
       b = -1
       for i in xrange(len(indx[0])):
           if i +1 == len(indx[0]) or indx[0][i+1] - indx[0][i] > 1:
               b = indx[0][i]
               if b - a <= 2:
                   a = indx[0][i+1]
                   continue
               self.firstSeg.append([a,b,self.hist[a:b]])
               if i +1 == len(indx[0]):
                  break
               else :
                   a = indx[0][i+1]

    def __findCrest__(self):
        self.__segByHist__()
        self.mean_len = 0
        for i in self.firstSeg:
            self.mean_len += i[1]-i[0]+1
        self.mean_len /= len(self.firstSeg)
        self.ans = []
        for i in self.firstSeg:
            if i[1] - i[0] +1 >= self.mean_len+self.mean_len*self.thirdThresh:
                 rect = self.bin_img[0:self.bin_img.shape[0],i[0]-1:i[1]+1]
                 fen = self.__cutBlock__(i[2],rect)
                 if len(fen) < 1:
                     self.ans.append(rect)
                     continue
                 for j in fen:
                    self.ans.append(j)
            else:
                self.ans.append(self.bin_img[0:self.bin_img.shape[0],i[0]-2:i[1]+2])

    def __cutBlock__(self,hist,img):
        mx = hist.max()
        mx *= self.fourThresh
        ans1 = []
        indx = np.where(hist>=mx)[0]
        if indx[len(indx)-1] - indx[0] < self.fiveThresh * self.mean_len or self.__calcuVariance__(hist) <= 2.0 :
            return ans1
        hist_ = []
        for i in xrange(1,len(indx),1):
            if indx[i] - indx[i-1] > self.mean_len * self.sixThresh:
                hist_.append([indx[i-1],indx[i]])
        ret = []
        for i in hist_:
            mi = np.min(hist[i[0]:i[1]])+1.
            indx2 = np.where(hist[i[0]:i[1]] <= mi)[0]
            len1 = len(indx2)/2
            ret.append(int(i[0]+indx2[len1]))
        last = 0
        for j in ret :
             tmp = img[0:img.shape[0],last: j]
             last = j
             ans1.append(tmp)
        tmp = img[0:img.shape[0],last:]
        ans1.append(tmp)
        return ans1
    def getSegs(self):
        self.__findCrest__()
        return self.ans

    def __calcuVariance__(self,hist):
        mean = np.mean(hist)
        sum = 0;
        for i in hist:
            sum += (i-mean) * (i-mean)
        sum /= len(hist)
        return  sum
    def showImg(self,img,name ='img'):
        cv.imshow(name,img)
        cv.waitKey(0)

img = cv.imread('imgs/7.jpg', cv.IMREAD_GRAYSCALE)
sg = SegTools(img)
ans = sg.getSegs()
for i in ans:
    sg.showImg(i)
