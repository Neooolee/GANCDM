# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 16:11:33 2019

@author: lijun
"""

from generator import *
import tensorflow as tf
from attention import *
import numpy as np

def attention_generator(img_x,img_y):#GAN-CDM-4

     with tf.variable_scope('ag'):

        xall=uattentionnet(image=img_x,reuse=False,name='attentionnet_x2y')
        yall=uattentionnet(image=img_y,reuse=True,name='attentionnet_x2y')
        xa=tf.nn.softmax(xall)
        ya=tf.nn.softmax(yall)

        
        fake_y=generator_unet(image=img_x,reuse=False,name='generatorx2y')

        fake_x_=fake_y*xa[:,:,:,1:2]+xa[:,:,:,0:1]
                    
        return fake_y,fake_x_,xa,ya,xall,yall

# def attention_generator(img_x,img_y):#GAN-CDM-6

#      with tf.variable_scope('ag'):

#         xall=uattentionnet(image=img_x,reuse=False,name='attentionnet_x2y')
#         yall=uattentionnet(image=img_y,reuse=True,name='attentionnet_x2y')
#         xa=tf.nn.softmax(xall)
#         ya=tf.nn.softmax(yall)

        
#         fake_y=generator_unet(image=img_x,reuse=False,name='generatorx2y')

#         fake_x_=fake_y*xa[:,:,:,1:2]+xa[:,:,:,0:1]
                    
#         return fake_y,fake_x_,xa,ya,xall,yall

# def attention_generator(img_x,img_y):#GAN-CDM-10

#      with tf.variable_scope('ag'):

#         xall=uattentionnet(image=img_x,reuse=False,name='attentionnet_x2y')
#         yall=uattentionnet(image=img_y,reuse=True,name='attentionnet_x2y')

#         xa=tf.nn.softmax(xall)
#         ya=tf.nn.softmax(yall)        
#         fake_y=generator_unet(image=img_x,reuse=False,name='generatorx2y')
#         f_x=tf.ones_like(fake_y)
#         fake_x_8=fake_y[:,:,:,:8]*xa[:,:,:,1:2]+f_x[:,:,:,:8]*xa[:,:,:,0:1]
#         fake_x_2=fake_y[:,:,:,8:]*(tf.ones_like(xa[:,:,:,0:1])-xa[:,:,:,0:1])
#         fake_x_=tf.concat([fake_x_8,fake_x_2],axis=-1)
                    
#         return fake_y,fake_x_,xa,ya,xall,yall

# def attention_generator(img_x,img_y):#GAN-CDM-8T

#      with tf.variable_scope('ag'):

#         xall=uattentionnet(image=img_x,reuse=False,name='attentionnet_x2y')
#         yall=uattentionnet(image=img_y,reuse=True,name='attentionnet_x2y')
#         xa=tf.nn.softmax(xall)
#         ya=tf.nn.softmax(yall)
        
#         fake_y=generator_unet(image=img_x,reuse=False,name='generatorx2y')

#         f_x=tf.ones_like(fake_y)
#         fake_x_6=fake_y[:,:,:,:6]*xa[:,:,:,1:2]+f_x[:,:,:,:6]*xa[:,:,:,0:1]
#         fake_x_2=fake_y[:,:,:,6:]*(tf.ones_like(xa[:,:,:,0:1])-xa[:,:,:,0:1])
#         fake_x_=tf.concat([fake_x_6,fake_x_2],axis=-1)
                    
#         return fake_y,fake_x_,xa,ya,xall,yall
       

        
