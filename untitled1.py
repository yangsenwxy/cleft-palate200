# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 15:15:22 2017

@author: Administrator
"""

import scipy.io as sio 
import numpy as np
import os
#
#for i in range(100):
path = r'E:\ysdeeplearn\deepcode\deeptrain\mfcc\cross1'
dirs = os.listdir( path )
l1='test_label.mat'
l2='train_label.mat'
l3='test_data.mat'
l4='train_data.mat'
path1 = r'E:\ysdeeplearn\deepcode\deeptrain\mfcc\cross1'
s1=np.load(r"E:\deepdata\36.npy")
s2=np.load(r"E:\deepdata\361.npy")
for s in dirs :
    newDir=os.path.join(path,s) 
#    newDir=path1+s
#    p=''
#    
#    newDir1=newDir+P+l1 
    newDir1=os.path.join(newDir,l1) 
    newDir2=os.path.join(newDir,l2) 
    newDir3=os.path.join(newDir,l3) 
    newDir4=os.path.join(newDir,l4) 
    
    

    load_test_label = r'E:\deeptrain\test_label826.mat'
    load_data = sio.loadmat(newDir1)
    load_matrix = load_data['test_label'] #假设文件中存有字符变量是matrix，例如matlab中save(load_fn, 'matrix');当然可以保存多个save(load_fn, 'matrix_x', 'matrix_y', ...);
    test_label=np.array(load_matrix,dtype='float32')




    load_train_label= r'E:\deeptrain\train_label827.mat'
    load_data = sio.loadmat(newDir2)
    load_matrix = load_data['train_label'] #假设文件中存有字符变量是matrix，例如matlab中save(load_fn, 'matrix');当然可以保存多个save(load_fn, 'matrix_x', 'matrix_y', ...);
    train_label=np.array(load_matrix,dtype='float32')

    load_train_data= r'E:\deeptrain\train_data826.mat'
    load_data = sio.loadmat(newDir4)
    load_matrix = load_data['train_data'] #假设文件中存有字符变量是matrix，例如matlab中save(load_fn, 'matrix');当然可以保存多个save(load_fn, 'matrix_x', 'matrix_y', ...);
    train_data=np.array(load_matrix,dtype='float32')

    load_test_data= r'E:\deeptrain\test_data826.mat'
    load_data = sio.loadmat(newDir3)
    load_matrix = load_data['test_data'] #假设文件中存有字符变量是matrix，例如matlab中save(load_fn, 'matrix');当然可以保存多个save(load_fn, 'matrix_x', 'matrix_y', ...);
    test_data=np.array(load_matrix,dtype='float32')
#    keep_prob = tf.placeholder(tf.float32)
    pathsave=r'E:\deepdata'
    v1='four.npy'
    v2='one.npy'
    new1=os.path.join(pathsave,s) 
#    new1=pathsave+s
    new2=new1+v1
    new3=new1+v2
#    pathsave1=r"E:\deepdata\"
    new1=os.path.join(pathsave,s) 
#    new2=os.path.join(new1,v1) 
#    new3=os.path.join(new1,v2)              
    np.save(new2,s1)
    np.save(new3,s2)