# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 16:07:54 2018

@author: Neoooli
"""

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
import math
import tensorflow.contrib.slim as slim
#构造可训练参数
def make_var(name, shape, trainable = True):
    return tf.get_variable(name, shape, trainable = trainable)
 
#定义卷积层
def conv2d(input_, output_dim, kernel_size=3, stride=2, padding = "SAME", name = "conv2d", biased = False):
    input_dim = input_.get_shape()[-1]
    with tf.variable_scope(name):
        kernel = make_var(name = 'weights', shape=[kernel_size, kernel_size, input_dim, output_dim])
        output = tf.nn.conv2d(input_, kernel, [1, stride, stride, 1], padding = padding)
        if biased:
            biases = make_var(name = 'biases', shape = [output_dim])
            output = tf.nn.bias_add(output, biases)
        return output     
#定义反卷积层
def deconv2d(input_, output_dim, kernel_size=3, stride=2, padding = "SAME", name = "deconv2d"):
    input_dim = input_.get_shape()[-1]
    batchsize=int(input_.get_shape()[0])
    input_height = int(input_.get_shape()[1])
    input_width = int(input_.get_shape()[2])
    with tf.variable_scope(name):
        kernel = make_var(name = 'weights', shape = [kernel_size, kernel_size, output_dim, input_dim])
        output = tf.nn.conv2d_transpose(input_, kernel, [batchsize, input_height * 2, input_width * 2, output_dim], [1, 2, 2, 1], padding = "SAME")
        return output

#定义空洞卷积层
def atrous_conv2d(input_, output_dim, kernel_size, dilation, padding = "SAME", name = "atrous_conv2d", biased = False):
    input_dim = input_.get_shape()[-1]
    with tf.variable_scope(name):
        kernel = make_var(name = 'weights', shape = [kernel_size, kernel_size, input_dim, output_dim])
        output = tf.nn.atrous_conv2d(input_, kernel, dilation, padding = padding)
        if biased:
            biases = make_var(name = 'biases', shape = [output_dim])
            output = tf.nn.bias_add(output, biases)
        return output
    
#定义全连接层
def fc_op(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [input_.get_shape()[-1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias 
def batch_norms(input_, name="batch_norm"):
    return tf.contrib.layers.batch_norm(input_, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope=name)
def instance_norms(input_, name="instance_norm"):
    return tf.contrib.layers.instance_norm(input_,scope=name)
def group_norms(x,name="group_norm"):
    with tf.variable_scope(name):
    # x_shape:[N,H, W,c]
        G=16
        N,H,W,C=x.get_shape()
        
        eps = 1e-5
        gamma = tf.get_variable("gamma", shape = [1,1,1,C], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        beta = tf.get_variable("beta", shape = [1,1,1,C], initializer=tf.constant_initializer(0.0))
        x = tf.reshape(x, [N,H,W,G,C//G])
        x_mean,x_var = tf.nn.moments(x,[1,2,4],keep_dims=True)
        x_normalized = (x - x_mean) / tf.sqrt(x_var + eps)
        results=tf.reshape(x_normalized,[N,H,W,C])
        results = gamma * results + beta
    
        return results
def instance_norm(input_, name="instance_norm"):
    return instance_norms(input_,name=name)
#    return batch_norms(input_,name=name)
#    return group_norms(input_,name=name)
#定义最大池化层
def max_pooling(input_, kernel_size, stride, name, padding = "SAME"):
    return tf.nn.max_pool(input_, ksize=[1, kernel_size, kernel_size, 1], strides=[1, stride, stride, 1], padding=padding, name=name)
#定义平均池化层
def avg_pooling(input_, kernel_size, stride, name, padding = "SAME"):
    return tf.nn.avg_pool(input_, ksize=[1, kernel_size, kernel_size, 1], strides=[1, stride, stride, 1], padding=padding, name=name)
 
#定义lrelu激活层
def lrelu(x, leak=0.2, name = "lrelu"):
    return tf.maximum(x, leak*x)
 
#定义relu激活层
def relu(input_, name = "relu"):
    return tf.nn.relu(input_, name = name)


#定义残差块
def residule_block_33(input_, output_dim, kernel_size = 3, stride = 1, dilation = 2, atrous = False, name = "res"):
    if atrous:
        conv2dc0 = atrous_conv2d(input_ = input_, output_dim = output_dim, kernel_size = kernel_size, dilation = dilation, name = (name + '_c0'))
        conv2dc0_norm = instance_norm(input_ = conv2dc0, name = (name + '_bn0'))
        conv2dc0_relu = relu(input_ = conv2dc0_norm)
        conv2dc1 = atrous_conv2d(input_ = conv2dc0_relu, output_dim = output_dim, kernel_size = kernel_size, dilation = dilation, name = (name + '_c1'))
        conv2dc1_norm = instance_norm(input_ = conv2dc1, name = (name + '_bn1'))
    else:
        conv2dc0 = conv2d(input_ = input_, output_dim = output_dim, kernel_size = kernel_size, stride = stride, name = (name + '_c0'))
        conv2dc0_norm = instance_norm(input_ = conv2dc0, name = (name + '_bn0'))
        conv2dc0_relu = relu(input_ = conv2dc0_norm)
        conv2dc1 = conv2d(input_ = conv2dc0_relu, output_dim = output_dim, kernel_size = kernel_size, stride = stride, name = (name + '_c1'))
        conv2dc1_norm = instance_norm(input_ = conv2dc1, name = (name + '_bn1'))
    add_raw = input_ + conv2dc1_norm
    output = relu(input_ = add_raw)
    return output
# 定义卷积注意力层
def Rk(input_, output_dim,  reuse=False, name=None):
  """ A residual block that contains two 3x3 convolutional layers
      with the same number of filters on both layer
  Args:
    input: 4D Tensor
    k: integer, number of filters (output depth)
    reuse: boolean
    name: string
  Returns:
    4D tensor (same shape as input)
  """
  with tf.variable_scope(name, reuse=reuse):
    with tf.variable_scope('layer1', reuse=reuse):
      
      padded1 = tf.pad(input_, [[0,0],[1,1],[1,1],[0,0]], 'REFLECT')
      conv1 = conv2d(input_ = padded1, output_dim = output_dim, kernel_size = 3, stride = 1,padding='VALID', name = (name + '_c1'))
      normalized1 = instance_norm(input_ = conv1, name = (name + '_bn1'))
      relu1 = tf.nn.relu(normalized1)

    with tf.variable_scope('layer2', reuse=reuse):

      padded2 = tf.pad(relu1, [[0,0],[1,1],[1,1],[0,0]], 'REFLECT')
      conv2 = conv2d(input_ = padded2, output_dim = output_dim, kernel_size = 3, stride = 1,padding='VALID', name = (name + '_c2'))
      normalized2 = instance_norm(input_ = conv2, name = (name + '_bn2'))
      
    output = input_+normalized2
    return output
def n_res_blocks(input_, reuse=False, n=6):
  depth = input.get_shape()[3]
  for i in range(1,n+1):
    output = Rk(input_, depth, reuse,'R{}_{}'.format(depth, i))
    input_ = output
  return output

def diydecay(steps,baselr):
    decay_steps = 100
    decay_rate=0.96
    cycle_step=100000
    n=steps//cycle_step
    clr=baselr*(0.8**n)
    
    steps=steps-n*cycle_step
    k=steps//decay_steps
    i=(-1)**k
    step=((i+1)/2)*steps-i*((k+1)//2)*decay_steps
     
    lr=tf.train.exponential_decay(clr,step,decay_steps,decay_rate, staircase=False)              
    return lr
def decay(global_steps,baselr):
    start_decay_step = 100000
    lr=tf.where(tf.greater_equal(global_steps,start_decay_step),
                diydecay(global_steps-start_decay_step,baselr),
                baselr)
    return lr

def grad(src):
    g_src_x = src[:, 1:, :, :] - src[:, :-1, :, :]
    g_src_y = src[:, :, 1:, :] - src[:, :, :-1, :]
    return g_src_x,g_src_y
def all_comp(grad1,grad2):
    v=[]
    dim1=grad1.shape[-1]
    dim2=grad2.shape[-1]
    for i in range(dim1):
        for j in range(dim2):
            v.append(tf.reduce_mean(((grad1[:,:,:,i]**2)*(grad2[:,:,:,j]**2)+1e-20))**0.25)
    return v

def get_grad(src,dst,level):
    gradx_loss=[]
    grady_loss=[]
    for i in range(level):
        gradx1,grady1=grad(src)
        gradx2,grady2=grad(dst)
        # lambdax2=2.0*tf.reduce_mean(tf.abs(gradx1)+1e-10)/tf.reduce_mean(tf.abs(gradx2)+1e-10)
        # lambday2=2.0*tf.reduce_mean(tf.abs(grady1)+1e-10)/tf.reduce_mean(tf.abs(grady2)+1e-10)
        lambdax2=1
        lambday2=1
        gradx1_s=(tf.nn.sigmoid(gradx1)*2)-1
        grady1_s=(tf.nn.sigmoid(grady1)*2)-1
        gradx2_s=(tf.nn.sigmoid(gradx2*lambdax2)*2)-1
        grady2_s=(tf.nn.sigmoid(grady2*lambday2)*2)-1
        gradx_loss+=all_comp(gradx1_s,gradx2_s)
        grady_loss+=all_comp(grady1_s,grady2_s)
        src=tf.keras.layers.AveragePooling2D((2,2),2,'same')(src)
        dst=tf.keras.layers.AveragePooling2D((2,2),2,'same')(dst)
    return gradx_loss,grady_loss

def exlusion_loss(src,dst,level=3):
    dim1=tf.cast(src.get_shape()[-1],dtype=tf.float32)
    dim2=tf.cast(dst.get_shape()[-1],dtype=tf.float32)
    gradx_loss,grady_loss=get_grad(src,dst,level)
    loss_gradxy=tf.reduce_sum(sum(gradx_loss)/(level*dim1*dim2))+tf.reduce_sum(sum(grady_loss)/(level*dim1*dim2))
    return loss_gradxy/2.0
