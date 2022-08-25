# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 15:13:17 2018

@author: lijun
"""

import numpy as np
from gdaldiy import *
def liner_2(input_):#2%线性拉伸,返回0~1之间的值
    def strech(img):
        low,high=np.percentile(img,(2,98))
        img[low>img]=low
        img[img>high]=high
        return (img-low)/(high-low+1e-10)
    if len(input_.shape)>2:
        for i in range(input_.shape[-1]):
            input_[:,:,i]=strech(input_[:,:,i])
    else:
        input_=strech(input_)    
    return input_
def randomflip(input_,n):
    #生成-3到2的随机整数，-1顺时针90度，-2顺时针180，-3顺时针270,0垂直翻转，1水平翻转，2不变
    if n<0:
        return np.rot90(input_,n)
    elif -1<n<2:
        return np.flip(input_,n)
    else: 
        return input_
def read_img(datapath,scale=65535):
    img=imgread(datapath)
    img[img>scale]=scale
    img=img/scale   
    return img

def read_imgs(datapath,scale=65535,k=2):
    img_list=[]
    l=len(datapath)
    for i in range(l):
        img=read_img(datapath[i],scale)
        img = randomflip(img,k)
        img=img[np.newaxis,:]
        img_list.append(img)    
    imgs=np.concatenate(img_list,axis=0)
    return imgs



