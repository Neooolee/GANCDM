# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 16:56:44 2018

@author: lijun
"""
import tensorflow as tf
from allnet import *

def generator(image, gf_dim=64, reuse=False, name="generator"): 
    #生成器输入尺度: 1*256*256*3
    output_dims=image.get_shape()[-1]  
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False        
        #第1个卷积模块，输出尺度: 1*256*256*64 
        c0 = relu(instance_norm(conv2d(input_ = image, output_dim = gf_dim, kernel_size = 7, stride = 1,name = 'g_e0_c'), name = 'g_e0_bn'))
        #第2个卷积模块，输出尺度: 1*128*128*128
        c1 = relu(instance_norm(conv2d(input_ = c0, output_dim = gf_dim * 2, kernel_size = 3, stride = 2, name = 'g_e1_c'), name = 'g_e1_bn'))
        #第3个卷积模块，输出尺度: 1*64*64*256
        c2 = relu(instance_norm(conv2d(input_ = c1, output_dim = gf_dim * 4, kernel_size = 3, stride = 2, name = 'g_e2_c'), name = 'g_e2_bn'))
        
        #9个残差块:
        r1 = residule_block_33(input_ = c2, output_dim = gf_dim*4, atrous = False, name='g_r1')
        r2 = residule_block_33(input_ = r1, output_dim = gf_dim*4, atrous = False, name='g_r2')
        r3 = residule_block_33(input_ = r2, output_dim = gf_dim*4, atrous = False, name='g_r3')
        r4 = residule_block_33(input_ = r3, output_dim = gf_dim*4, atrous = False, name='g_r4')
        r5 = residule_block_33(input_ = r4, output_dim = gf_dim*4, atrous = False, name='g_r5')
        r6 = residule_block_33(input_ = r5, output_dim = gf_dim*4, atrous = False, name='g_r6')
        r7 = residule_block_33(input_ = r6, output_dim = gf_dim*4, atrous = False, name='g_r7')
        r8 = residule_block_33(input_ = r7, output_dim = gf_dim*4, atrous = False, name='g_r8')
        r9 = residule_block_33(input_ = r8, output_dim = gf_dim*4, atrous = False, name='g_r9')
       
        #第9个残差块的输出尺度: 1*64*64*256
 
		#第1个反卷积模块，输出尺度: 1*128*128*128
        d1 = relu(instance_norm(deconv2d(input_ = r9, output_dim = gf_dim * 2, kernel_size = 3, stride = 2, name = 'g_d1_dc'),name = 'g_d1_bn'))
		#第2个反卷积模块，输出尺度: 1*256*256*64
        d2 = relu(instance_norm(deconv2d(input_ = d1, output_dim = gf_dim, kernel_size = 3, stride = 2, name = 'g_d2_dc'),name = 'g_d2_bn'))
		#最后一个卷积模块，输出尺度: 1*256*256*3
        d3 = conv2d(input_=d2, output_dim  = output_dims, kernel_size = 7, stride = 1,name = 'g_d3_c')
		#经过tanh函数激活得到生成的输出
        output = tf.nn.sigmoid(d3)
        return output
def generator_resnet(image, gf_dim=64, reuse=False, name="generator"):

    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        def residule_block(input_, dim, ks=3, s=1, name='res'):
            p = int((ks - 1) / 2)
            pad1 = tf.pad(input_, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            conv1 = instance_norm(conv2d(pad1, dim, ks, s, padding='VALID', name=name+'_c1'), name+'_bn1')
            pad2 = tf.pad(tf.nn.relu(conv1), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            conv2 = instance_norm(conv2d(pad2, dim, ks, s, padding='VALID', name=name+'_c2'), name+'_bn2')
            return input_ + conv2

        # Justin Johnson's model from https://github.com/jcjohnson/fast-neural-style/
        # The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128,
        # R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3
        c0 = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        c1 = tf.nn.relu(instance_norm(conv2d(c0, gf_dim, 7, 1, padding='VALID', name='g_e1_c'), 'g_e1_bn'))
        c2 = tf.nn.relu(instance_norm(conv2d(c1, gf_dim*2, 3, 2, name='g_e2_c'), 'g_e2_bn'))
        c3 = tf.nn.relu(instance_norm(conv2d(c2, gf_dim*4, 3, 2, name='g_e3_c'), 'g_e3_bn'))
        # define G network with 9 resnet blocks
#        r9=n_res_blocks(input_=c3, reuse=False, n=9)
        r1 = residule_block(c3, gf_dim*4, name='g_r1')
        r2 = residule_block(r1, gf_dim*4, name='g_r2')
        r3 = residule_block(r2, gf_dim*4, name='g_r3')
        r4 = residule_block(r3, gf_dim*4, name='g_r4')
        r5 = residule_block(r4, gf_dim*4, name='g_r5')
        r6 = residule_block(r5, gf_dim*4, name='g_r6')
        r7 = residule_block(r6, gf_dim*4, name='g_r7')
        r8 = residule_block(r7, gf_dim*4, name='g_r8')
        r9 = residule_block(r8, gf_dim*4, name='g_r9')

        d1 = deconv2d(r9, gf_dim*2, 3, 2, name='g_d1_dc')
        d1 = tf.nn.relu(instance_norm(d1, 'g_d1_bn'))
        d2 = deconv2d(d1, gf_dim, 3, 2, name='g_d2_dc')
        d2 = tf.nn.relu(instance_norm(d2, 'g_d2_bn'))
        d2 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        pred = tf.nn.tanh(conv2d(d2, 3, 7, 1, padding='VALID', name='g_pred_c'))

        return pred

def generator_unet(image, gf_dim=64, reuse=False, name="generator"):
    output_dim=image.get_shape()[-1]
    # dropout_rate = 0.8
    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

               # image is (256 x 256 x input_c_dim)
        e0 = lrelu(conv2d(image,gf_dim,stride=1,name='g_e0_conv'))
        # e1 is (128 x 128 x self.gf_dim)
        e1 = lrelu(instance_norm(conv2d(e0, gf_dim*2, name='g_e1_conv'),'g_bn_e1'))
        e2 = lrelu(instance_norm(conv2d(e1, gf_dim*4, name='g_e2_conv'), 'g_bn_e2'))
        # e2 is (64 x 64 x self.gf_dim*2)
        e3 = lrelu(instance_norm(conv2d(e2, gf_dim*8, name='g_e3_conv'), 'g_bn_e3'))
        # e3 is (32 x 32 x self.gf_dim*4)
        e4 = lrelu(instance_norm(conv2d(e3, gf_dim*8, name='g_e4_conv'), 'g_bn_e4'))
        # e4 is (16 x 16 x self.gf_dim*8)
        e5 = lrelu(instance_norm(conv2d(e4, gf_dim*8, name='g_e5_conv'), 'g_bn_e5'))
        # e5 is (8 x 8 x self.gf_dim*8)
        e6 = lrelu(instance_norm(conv2d(e5, gf_dim*8, name='g_e6_conv'), 'g_bn_e6'))
        # e6 is (4 x 4 x self.gf_dim*8)
      
        d1 = tf.nn.relu(instance_norm(deconv2d(e6, gf_dim*8, name='g_d1'),'d_bn_d0'))
#        d1 = tf.nn.dropout(d1, dropout_rate)
        d1 = tf.concat([d1, e5],3)
        # d1 is (8 x 8 x self.gf_dim*8*2)

        d2 = tf.nn.relu(instance_norm(deconv2d(d1, gf_dim*8, name='g_d2'),'g_bn_d1'))
#        d2 = tf.nn.dropout(d2, dropout_rate)
        d2 = tf.concat([d2,e4], 3)
        # d2 is (16 x 16 x self.gf_dim*8*2)

        d3 = tf.nn.relu(instance_norm(deconv2d(d2, gf_dim*8, name='g_d3'),'g_bn_d2'))
#        d3 = tf.nn.dropout(d3, dropout_rate)
        d3 = tf.concat([d3, e3], 3)
        # d3 is (32 x 32 x self.gf_dim*8*2)

        d4 = tf.nn.relu(instance_norm(deconv2d(d3, gf_dim*4, name='g_d4'),'g_bn_d3'))
        d4 = tf.concat([d4,e2], 3)
        # d4 is (16 x 16 x self.gf_dim*8*2)

        d5 = tf.nn.relu(instance_norm(deconv2d(d4, gf_dim*2, name='g_d5'),'g_bn_d4'))
        d5 = tf.concat([d5,e1], 3)
        # d5 is (32 x 32 x self.gf_dim*4*2)

        d6 = tf.nn.relu(instance_norm(deconv2d(d5,gf_dim,name='g_d6'),'g_bn_d5'))
        d6 = tf.concat([d6,e0],axis=3)
        # d8 is (256 x 256 x output_c_dim)
        d6 = conv2d(d6,output_dim, kernel_size = 3, stride = 1,name = 'out_conv')

        return tf.nn.sigmoid(d6)

