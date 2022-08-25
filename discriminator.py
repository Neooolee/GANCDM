# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 16:58:01 2018

@author: lijun
"""
import tensorflow as tf
from allnet import *


def cnodiscriminator(image,reuse=False, name="discriminator"):
    df_dim=64
    with tf.variable_scope(name):

        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        h0 = lrelu(conv2d(input_=image,output_dim=df_dim,kernel_size=4,stride=2,name='layer1-conv2d',biased=False))

        h1 = lrelu(instance_norm(conv2d(input_=h0,output_dim=df_dim*2,kernel_size=4,stride=2,name='layer2-conv2d',biased=False), 'd_bn1'))

        h2 = lrelu(instance_norm(conv2d(input_=h1,output_dim=df_dim*4,kernel_size=4,stride=2,name='layer3-conv2d',biased=False), 'd_bn2'))

        h3 = lrelu(instance_norm(conv2d(input_=h2,output_dim=df_dim*8,kernel_size=4,stride=2,name='layer4-conv2d',biased=False), 'd_bn3'))

        patch = conv2d(input_=h3,output_dim=1,kernel_size=4,stride=1,name='layer5-conv2d',biased=False)

        return patch

