# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 16:57:36 2018

@author: Neoooli
"""

from __future__ import print_function
 
import argparse
from datetime import datetime
from random import shuffle
import random
import os
import sys
import time
import math
import tensorflow as tf
import numpy as np
import glob
from PIL import Image
 
from batch_read_img import *
from generator import *
from discriminator import *
from attention import *
from attention_generator import *
parser = argparse.ArgumentParser(description='')
 
parser.add_argument("--snapshot_dir", default='./snapshots/', help="path of snapshots") #保存模型的路径
parser.add_argument("--out_dir", default='./train_out', help="path of train outputs") #训练时保存可视化输出的路径
parser.add_argument("--image_size", type=int, default=384, help="load image size") #网络输入的尺度
parser.add_argument("--random_seed", type=int, default=1234, help="random seed") #随机数种子
parser.add_argument('--base_lr', type=float, default=0.0002, help='initial learning rate for adam') #基础学习率
parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='# of epoch') #训练的epoch数量
parser.add_argument('--epoch_step', dest='epoch_step', type=int, default=100, help='# of epoch to decay lr') #训练中保持学习率不变的epoch数量
parser.add_argument("--lamda", type=float, default=10.0, help="L1 lamda") #训练中L1_Loss前的乘数
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam') #adam优化器的beta1参数
parser.add_argument('--beta2', dest='beta2', type=float, default=0.9, help='momentum term of adam') #adam优化器的beta1参数
parser.add_argument("--summary_pred_every", type=int, default=200, help="times to summary.") #训练中每过多少step保存训练日志(记录一下loss值)
parser.add_argument("--write_pred_every", type=int, default=1000, help="times to write.") #训练中每过多少step保存可视化结果
parser.add_argument("--save_pred_every", type=int, default=100000, help="times to save.") #训练中每过多少step保存模型(可训练参数)
parser.add_argument("--x_train_data_path", default='/dat01/wuzhaocong/data/clouddetection/WHUL8-CDGAN/cloudDNclips/', help="path of x training datas.") #x域的训练图片路径
parser.add_argument("--y_train_data_path", default='/dat01/wuzhaocong/data/clouddetection/WHUL8-CDGAN/clearDNclips/', help="path of y training datas.") #y域的训练图片路径
parser.add_argument("--z_train_data_path", default='/dat01/wuzhaocong/data/clouddetection/WHUL8-CDGAN/allcloudDNclips/', help="path of z training datas.") #y域的训练图片路径
parser.add_argument("--batch_size", type=int, default=1, help="load batch size") #batch_size
parser.add_argument("--d_lambda", type=int, default=2, help="load batch size") #batch_size
args = parser.parse_args()
 
def save(saver, sess, logdir, step): #保存模型的save函数
   model_name = 'model' #保存的模型名前缀
   checkpoint_path = os.path.join(logdir, model_name) #模型的保存路径与名称
   if not os.path.exists(logdir): #如果路径不存在即创建
      os.makedirs(logdir)
   saver.save(sess, checkpoint_path, global_step=step) #保存模型
   print('The checkpoint has been created.')

def get_write_picture(row_list): #get_write_picture函数得到训练过程中的可视化结果
    row_=[]    
    for i in range(len(row_list)):
        row=row_list[i] 
        col_=[]
        for image in row:
            x_image=image[:,:,[2,1,0]]
            if i<1:
                x_image=liner_2(x_image)
            col_.append(x_image)
        row_.append(np.concatenate(col_,axis=1))
    if len(row_list)==1:
        output = np.concatenate(col_,axis=1)
    else:
        output = np.concatenate(row_, axis=0) #得到训练中可视化结果
    return output*255
 
def make_train_data_list(data_path): #make_train_data_list函数得到训练中的x域和y域的图像路径名称列表
    filepath= glob.glob(os.path.join(data_path, "*")) #读取全部的x域图像路径名称列表
    image_path_lists=[]
    for i in range(len(filepath)):
         path=glob.glob(os.path.join(filepath[i], "*"))
         for j in range(len(path)):
             image_path_lists.append(path[j]) #将x域图像数量与y域图像数量对齐
    return image_path_lists
    
def l1_loss(src, dst): #定义l1_loss
    return tf.reduce_mean(tf.abs(src - dst))
 
def gan_loss(src, dst): #定义gan_loss，在这里用了二范数
    return tf.reduce_mean((src-dst)**2)

def main():
    tf.reset_default_graph()
    if not os.path.exists(args.snapshot_dir): #如果保存模型参数的文件夹不存在则创建
        os.makedirs(args.snapshot_dir)
    if not os.path.exists(args.out_dir): #如果保存训练中可视化输出的文件夹不存在则创建
        os.makedirs(args.out_dir)
    y_datalists = make_train_data_list(args.y_train_data_path)
    z_datalists = make_train_data_list(args.z_train_data_path)
    x_datalists = make_train_data_list(args.x_train_data_path)+z_datalists #得到数量相同的x域和y域图像路径名称列表
    tf.set_random_seed(args.random_seed) #初始一下随机数
    x_img = tf.placeholder(tf.float32,shape=[args.batch_size, args.image_size, args.image_size,8],name='x_img') #输入的x域图像
    y_img = tf.placeholder(tf.float32,shape=[args.batch_size, args.image_size, args.image_size,8],name='y_img') #输入的y域图像

    fake_y,fake_x_,xall,yall,xlogits,ylogits=attention_generator(x_img,y_img)
    xr=xall[:,:,:,0:1]
    xt=xall[:,:,:,1:2]
    yr=yall[:,:,:,0:1]
    decay_step = tf.placeholder(tf.float32,name='step')
    max_r=1e-3
    patch_real = cnodiscriminator(image=y_img, reuse=False, name='discriminator_y') #判别器返回的对真实的y域图像的判别结果
    patch_fake = cnodiscriminator(image=fake_y, reuse=True, name='discriminator_y') #判别器返回的对生成的y域图像的判别结果

    kapa_r=tf.random.uniform(shape=[1,1,1,1],minval=0.0,maxval=max_r)
    kapa_a=tf.random.uniform(shape=[1,1,1,1],minval=0.0,maxval=max_r-kapa_r)
    kapa_t=tf.ones(shape=[1,1,1,1])-kapa_r-kapa_a            
    y_softlabel=tf.concat([kapa_r,kapa_t,kapa_a],axis=-1)*tf.ones_like(xr)

    opt_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_softlabel,logits=ylogits))
    e_loss=exlusion_loss(xr,fake_y)+exlusion_loss(xt,fake_y)
    a_loss=gan_loss(patch_fake, tf.ones_like(patch_fake))+args.lamda*l1_loss(x_img,fake_x_)+opt_loss+e_loss
    
    
    d_loss =(gan_loss(patch_real,tf.ones_like(patch_real))+gan_loss(patch_fake,tf.zeros_like(patch_fake)))/2

    xr=tf.concat([xr,xr,xr],-1) 
    xt=tf.concat([xt,xt,xt],-1)   
    yr=tf.concat([yr,yr,yr],-1)
    xa=tf.concat([xall[:,:,:,2:3],xall[:,:,:,2:3],xall[:,:,:,2:3]],-1) 

    tf.summary.scalar("dis_loss", d_loss) #记录判别器的loss的日志
    

    tf.summary.scalar("a_loss", a_loss) #记录生成器loss的日志

 
   
    d_vars = [v for v in tf.trainable_variables() if 'discriminator' in v.name] #所有判别器的可训练参数


    
    a_vars = [v for v in tf.trainable_variables() if 'ag' in v.name]
    
    global_step = tf.placeholder(tf.float32,name='step')
    learning_rate = tf.placeholder(tf.float32,name='lr')
    learningrate=decay(global_step,learning_rate)
    tf.summary.scalar('learning_rate', learningrate)

    a_optim = tf.train.AdamOptimizer(learningrate, beta1=args.beta1, name='Adam_AG').minimize(a_loss, var_list=a_vars)
    d_optim = tf.train.AdamOptimizer(learningrate, beta1=args.beta1, name='Adam_D').minimize(d_loss, var_list=d_vars) 

    train_op = tf.group(d_optim,a_optim)
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(args.snapshot_dir, graph=tf.get_default_graph())

    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=1)    
    config = tf.ConfigProto(gpu_options=gpu_options)
    
    config.gpu_options.allow_growth = True #设定显存不超量使用
    sess = tf.Session(config=config) #新建会话层
 
    saver = tf.train.Saver(max_to_keep=600) #模型保存器
    
    ckpt = tf.train.get_checkpoint_state(args.snapshot_dir)
    if ckpt and ckpt.model_checkpoint_path:
      meta_graph_path = ckpt.model_checkpoint_path + ".meta"
      restore = tf.train.import_meta_graph(meta_graph_path)
      restore.restore(sess, tf.train.latest_checkpoint(args.snapshot_dir))
      step = int(meta_graph_path.split("-")[1].split(".")[0])
    else:
      sess.run(tf.global_variables_initializer())
      step = 0
    lenx=len(x_datalists)
    leny=len(y_datalists)

    flops = tf.profiler.profile(tf.get_default_graph(), options=tf.profiler.ProfileOptionBuilder.float_operation())
    print('FLOPs: {}'.format(flops.total_float_ops))
    for epoch in range(args.epoch): #训练epoch数       
        shuffle(x_datalists)       #每训练一个epoch，就打乱一下x域图像顺序
        shuffle(y_datalists) #每训练一个epoch，就打乱一下y域图像顺序
        
        startx=0
        starty=0

        while (starty+args.batch_size)<=leny:
            k = np.random.randint(low=-3, high=3,size=1)             
            batch_x_image=read_imgs(x_datalists[startx:startx+args.batch_size],65535,k)[:,:,:,[1,2,3,4,5,6,8,9]]
            batch_y_image=read_imgs(y_datalists[starty:starty+args.batch_size],65535,k)[:,:,:,[1,2,3,4,5,6,8,9]] #读取x域图像和y域图像            

            feed_dict = {x_img : batch_x_image, y_img : batch_y_image,learning_rate:args.base_lr,global_step:step} #得到feed_dict     
            dl,al,_= sess.run([d_loss,a_loss,train_op], feed_dict=feed_dict) #得到每个step中的生成器和判别器loss
      
            step=step+1
#            k=k+1
            starty=starty+args.batch_size
            startx=startx+args.batch_size            
            if (startx+args.batch_size)>=lenx:
                shuffle(x_datalists)
                startx=0   

            if step% args.save_pred_every == 0: #每过save_pred_every次保存模型
                save(saver, sess, args.snapshot_dir, step)
            if step% args.summary_pred_every == 0: #每过summary_pred_every次保存训练日志
                summary= sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary, step)

            if step % args.write_pred_every == 0: #每过write_pred_every次写一下训练的可视化结果
                fake_y_value,ax,rx,ry,tx= sess.run([fake_y,xa,xr,yr,xt], feed_dict=feed_dict) #run出网络输出
                write_image = get_write_picture([[batch_x_image[0][:,:,[1,2,3]],fake_y_value[0][:,:,[1,2,3]],batch_y_image[0][:,:,[1,2,3]]],[rx[0],ry[0],tx[0]]]) #得到训练的可视化结果
                write_image_name = args.out_dir + "/out"+ str(epoch+1)+'_'+str(step)+ ".png" #待保存的训练可视化结果路径与名称
                Image.fromarray(np.uint8(write_image)).save(write_image_name) #保存训练的可视化结果
                print('epoch step       a_loss       d_loss   ')
                print('{:d}     {:d}    {:.3f}         {:.3f}   '.format(epoch+1, step,al,dl))
            if step==1000000:
                exit()
 
                
if __name__ == '__main__':
    main()
