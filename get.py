# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 10:18:09 2018

@author: Administrator
"""

import scipy.io as sio 
import numpy as np
import os
import random as rd

def get():
    load_train_label= r'E:\杨森 DNN 算特征\杨森DNN\读取成人5300个韵母数据\柳银代码\matlab程序 柳银 2016年5月9日 所有的代码\matlab程序2016年3月20日\order_true.mat'
    load_data = sio.loadmat(load_train_label)
    load_matrix = load_data['c'] #假设文件中存有字符变量是matrix，例如matlab中save(load_fn, 'matrix');当然可以保存多个save(load_fn, 'matrix_x', 'matrix_y', ...);
    train_label=np.array(load_matrix,dtype='float32')
    
    
    load_train_data= r'E:\杨森 DNN 算特征\杨森DNN\读取成人5300个韵母数据\柳银代码\matlab程序 柳银 2016年5月9日 所有的代码\matlab程序2016年3月20日\X_chilren_128.mat'
    load_data = sio.loadmat(load_train_data)
    load_matrix = load_data['X_chilren_128'] #假设文件中存有字符变量是matrix，例如matlab中save(load_fn, 'matrix');当然可以保存多个save(load_fn, 'matrix_x', 'matrix_y', ...);
    train_data=np.array(load_matrix,dtype='float32')
    
    
    
    load_test_label = r'E:\杨森 DNN 算特征\杨森DNN\读取成人5300个韵母数据\柳银代码\matlab程序 柳银 2016年5月9日 所有的代码\matlab程序2016年3月20日\label_true.mat'
    load_data = sio.loadmat(load_test_label)
    load_matrix = load_data['b'] #假设文件中存有字符变量是matrix，例如matlab中save(load_fn, 'matrix');当然可以保存多个save(load_fn, 'matrix_x', 'matrix_y', ...);
    test_label1=np.array(load_matrix,dtype='float32')
    
    n=np.shape(test_label1)[1]
    
    test_label=np.zeros((n,2))
    
    for i in range(n):
        if test_label1[0,i]==1:
            test_label[i,:]=[1 ,0]
        elif (test_label1[0,i] > 1):
             test_label[i,:]=[0 ,1]
            
    
    idx=np.where(train_label==1)[1]
    #idx=idx.reshape(1,58)
#    print(idx[-1])
    
    zui=np.max(train_data)
    
    train_data=train_data/zui
    #
    #
    test_data1=train_data[idx[0]:(idx[-1]+1),:]
    test_label2=test_label[idx[0]:(idx[-1]+1),:]                     
    
    train_data_copy=train_data
    test_label_copy=test_label
    
    train_data_shan=np.delete(train_data_copy, idx, axis=0)
    test_label_shan=np.delete(test_label_copy, idx, axis=0)
    
    l1=np.shape(train_data_shan)[0]
    #l2=len(l1)
    
    idx1=rd.sample(range(l1),l1) 
    
    train_data_shan1=train_data_shan[idx1,:]
    train_label_shan=test_label_shan[idx1,:]
    return train_data_shan1,train_label_shan,test_data1,test_label2

train_data,train_label,test_data,test_label=get()

#print(train_data[0,0])