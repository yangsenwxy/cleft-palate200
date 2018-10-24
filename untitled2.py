# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 18:01:41 2017

@author: Administrator
"""

import scipy.io as sio 
import numpy as np
import os
#
#for i in range(100):
path = "E:\shenduxuexidata\5300_2017_11_12LPCC\cross"
dirs = os.listdir( path )
l1='test_label.mat'
l2='trian_label.mat'
l3='test_data.mat'
l4='train_data.mat'

#load_test_label = r'E:\shenduxuexidata\5300_2017_11_12LPCC\cross\101\test_label.mat'
#load_data = sio.loadmat(load_test_label)
#load_matrix = load_data['test_label'] #假设文件中存有字符变量是matrix，例如matlab中save(load_fn, 'matrix');当然可以保存多个save(load_fn, 'matrix_x', 'matrix_y', ...);
#test_label=np.array(load_matrix,dtype='float32')
##

#for s in dirs :
#    newDir=os.path.join(path,s) 
#    
#    
#    newDir1=os.path.join(newDir,l1) 
#    newDir2=os.path.join(newDir,l2) 
#    newDir3=os.path.join(newDir,l3) 
#    newDir4=os.path.join(newDir,l4) 
#    
#    
#
#    load_test_label = r'E:\deeptrain\test_label826.mat'
#    load_data = sio.loadmat(l1)
#    load_matrix = load_data['test_label'] #假设文件中存有字符变量是matrix，例如matlab中save(load_fn, 'matrix');当然可以保存多个save(load_fn, 'matrix_x', 'matrix_y', ...);
#    test_label=np.array(load_matrix,dtype='float32')
#
#
#
#
#    load_train_label= r'E:\deeptrain\train_label827.mat'
#    load_data = sio.loadmat(l2)
#    load_matrix = load_data['train_label'] #假设文件中存有字符变量是matrix，例如matlab中save(load_fn, 'matrix');当然可以保存多个save(load_fn, 'matrix_x', 'matrix_y', ...);
#    train_label=np.array(load_matrix,dtype='float32')
#
#    load_train_data= r'E:\deeptrain\train_data826.mat'
#    load_data = sio.loadmat(l4)
#    load_matrix = load_data['train_data'] #假设文件中存有字符变量是matrix，例如matlab中save(load_fn, 'matrix');当然可以保存多个save(load_fn, 'matrix_x', 'matrix_y', ...);
#    train_data=np.array(load_matrix,dtype='float32')
#
#    load_test_data= r'E:\deeptrain\test_data826.mat'
#    load_data = sio.loadmat(l3)
#    load_matrix = load_data['test_data'] #假设文件中存有字符变量是matrix，例如matlab中save(load_fn, 'matrix');当然可以保存多个save(load_fn, 'matrix_x', 'matrix_y', ...);
#    test_data=np.array(load_matrix,dtype='float32')