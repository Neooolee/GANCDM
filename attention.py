# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 10:06:33 2018

@author: Neoooli
"""

import tensorflow as tf
from allnet import *
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

def uattentionnet(image, gf_dim=64, reuse=False, name="attentionnet"):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        e0 = lrelu(conv2d(image,gf_dim,stride=1,name='g_e0_conv'))

        e1 = lrelu(instance_norm(conv2d(e0, gf_dim*2, name='g_e1_conv'),'g_bn_e1'))
        e2 = lrelu(instance_norm(conv2d(e1, gf_dim*4, name='g_e2_conv'), 'g_bn_e2'))

        e3 = lrelu(instance_norm(conv2d(e2, gf_dim*8, name='g_e3_conv'), 'g_bn_e3'))

        e4 = lrelu(instance_norm(conv2d(e3, gf_dim*8, name='g_e4_conv'), 'g_bn_e4'))

        e5 = lrelu(instance_norm(conv2d(e4, gf_dim*8, name='g_e5_conv'), 'g_bn_e5'))

        e6 = lrelu(instance_norm(conv2d(e5, gf_dim*8, name='g_e6_conv'), 'g_bn_e6'))
      
        d1 = tf.nn.relu(instance_norm(deconv2d(e6, gf_dim*8, name='g_d1'),'d_bn_d0'))

        d1 = tf.concat([d1, e5],3)

        d2 = tf.nn.relu(instance_norm(deconv2d(d1, gf_dim*8, name='g_d2'),'g_bn_d1'))

        d2 = tf.concat([d2,e4], 3)

        d3 = tf.nn.relu(instance_norm(deconv2d(d2, gf_dim*8, name='g_d3'),'g_bn_d2'))

        d3 = tf.concat([d3, e3], 3)

        d4 = tf.nn.relu(instance_norm(deconv2d(d3, gf_dim*4, name='g_d4'),'g_bn_d3'))
        d4 = tf.concat([d4,e2], 3)

        d5 = tf.nn.relu(instance_norm(deconv2d(d4, gf_dim*2, name='g_d5'),'g_bn_d4'))
        d5 = tf.concat([d5,e1], 3)

        d6 = tf.nn.relu(instance_norm(deconv2d(d5,gf_dim,name='g_d6'),'g_bn_d5'))
        d6 = tf.concat([d6,e0],axis=3)

        d6 = conv2d(d6,3, kernel_size = 3, stride = 1,name = 'out_conv')

        return d6

