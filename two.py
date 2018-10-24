# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 21:16:46 2018

@author: Administrator
"""

from __future__ import print_function
import tensorflow as tf
import scipy.io as sio 
import numpy as np
s=np.array([])
s1=np.array([])
load_test_label = r'E:\ysdeeplearn\deepcode\deeptrain\mfcc\cross\49\test_label.mat'
load_data = sio.loadmat(load_test_label)
load_matrix = load_data['test_label'] #假设文件中存有字符变量是matrix，例如matlab中save(load_fn, 'matrix');当然可以保存多个save(load_fn, 'matrix_x', 'matrix_y', ...);
test_label=np.array(load_matrix,dtype='float32')


on_train = tf.placeholder(tf.bool)

load_train_label= r'E:\ysdeeplearn\deepcode\deeptrain\mfcc\cross\49\train_label.mat'
load_data = sio.loadmat(load_train_label)
load_matrix = load_data['train_label'] #假设文件中存有字符变量是matrix，例如matlab中save(load_fn, 'matrix');当然可以保存多个save(load_fn, 'matrix_x', 'matrix_y', ...);
train_label=np.array(load_matrix,dtype='float32')

load_train_data= r'E:\ysdeeplearn\deepcode\deeptrain\mfcc\cross\49\train_data.mat'
load_data = sio.loadmat(load_train_data)
load_matrix = load_data['train_data'] #假设文件中存有字符变量是matrix，例如matlab中save(load_fn, 'matrix');当然可以保存多个save(load_fn, 'matrix_x', 'matrix_y', ...);
train_data=np.array(load_matrix,dtype='float32')

load_test_data= r'E:\ysdeeplearn\deepcode\deeptrain\mfcc\cross\49\test_data.mat'
load_data = sio.loadmat(load_test_data)
load_matrix = load_data['test_data'] #假设文件中存有字符变量是matrix，例如matlab中save(load_fn, 'matrix');当然可以保存多个save(load_fn, 'matrix_x', 'matrix_y', ...);
test_data=np.array(load_matrix,dtype='float32')

f1=len(train_label)
train_label1=np.zeros((f1,2))


for i in range(f1):
    if train_label[i,0]==1:
        train_label1[i,:]=[1,0]
    else:
        train_label1[i,:]=[0,1]

        