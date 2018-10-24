# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 19:57:51 2017

@author: Administrator
"""

from __future__ import print_function
import tensorflow as tf
import scipy.io as sio 
import numpy as np

#load_test_label = r'E:\ysdeeplearn\deepcode\deeptrain\mfcc\85128sp\test_label.mat'
#load_data = sio.loadmat(load_test_label)
#load_matrix = load_data['test_label'] #假设文件中存有字符变量是matrix，例如matlab中save(load_fn, 'matrix');当然可以保存多个save(load_fn, 'matrix_x', 'matrix_y', ...);
#test_label=np.array(load_matrix,dtype='float32')
#
#
#
#
#load_train_label= r'E:\ysdeeplearn\deepcode\deeptrain\mfcc\85128sp\train_label.mat'
#load_data = sio.loadmat(load_train_label)
#load_matrix = load_data['train_label'] #假设文件中存有字符变量是matrix，例如matlab中save(load_fn, 'matrix');当然可以保存多个save(load_fn, 'matrix_x', 'matrix_y', ...);
#train_label=np.array(load_matrix,dtype='float32')
#
#load_train_data= r'E:\ysdeeplearn\deepcode\deeptrain\mfcc\85128sp\train_data.mat'
#load_data = sio.loadmat(load_train_data)
#load_matrix = load_data['train_data'] #假设文件中存有字符变量是matrix，例如matlab中save(load_fn, 'matrix');当然可以保存多个save(load_fn, 'matrix_x', 'matrix_y', ...);
#train_data=np.array(load_matrix,dtype='float32')
#
#load_test_data= r'E:\ysdeeplearn\deepcode\deeptrain\mfcc\85128sp\test_data.mat'
#load_data = sio.loadmat(load_test_data)
#load_matrix = load_data['test_data'] #假设文件中存有字符变量是matrix，例如matlab中save(load_fn, 'matrix');当然可以保存多个save(load_fn, 'matrix_x', 'matrix_y', ...);
#test_data=np.array(load_matrix,dtype='float32')
np.save(r"E:\deepdata\21.npy",s)
np.save(r"E:\deepdata\211.npy",s1)