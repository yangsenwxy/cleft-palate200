# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 21:51:12 2018

@author: Administrator
"""

from tensorflow.contrib import rnn
#from sklearn import preprocessing
#from run_rnn_demo10 import run_rnn_demo10
#from func_write_data_to_mat import func_write_data_to_mat

#from read_mat_to_list import read_mat_to_list
#from list_extend_to_list import list_extend_to_list
#from list_length import list_length
from sklearn.model_selection import KFold
import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops.rnn_cell_impl import DropoutWrapper
import scipy.io as sio 
import numpy as np
import random as rd
import os
#import h5py
#from random_choice_list import random_choice_list
path="E:\ysdeeplearn\词rf\\"
p='s.npy'
p1='s1.npy'



def get():
#    load_train_label= r'D:\ys\外星人\yumu\order_vocal_area.mat'
#    load_data = sio.loadmat(load_train_label)
#    load_matrix = load_data['order_l'] #假设文件中存有字符变量是matrix，例如matlab中save(load_fn, 'matrix');当然可以保存多个save(load_fn, 'matrix_x', 'matrix_y', ...);
#    train_label=np.array(load_matrix,dtype='float32')
    
    
    load_train_data=r'E:\ysdeeplearn\deepcode\deeptrain\何飞\yumu\data_burg.mat'
#    load_data = h5py.File(load_train_data)
    load_data = sio.loadmat(load_train_data)
    load_matrix = load_data['data_l'] #假设文件中存有字符变量是matrix，例如matlab中save(load_fn, 'matrix');当然可以保存多个save(load_fn, 'matrix_x', 'matrix_y', ...);
    train_data=np.array(load_matrix,dtype='float32')
#    train_data =np.transpose(train_data)
#    train_data=np.load(r'E:\ysdeeplearn\deepcode\deeptrain\何飞\yumu\train_data.npy')
    
    
    
    load_test_label = r'E:\ysdeeplearn\deepcode\deeptrain\何飞\yumu\label_burg.mat'
    load_data = sio.loadmat(load_test_label)
    load_matrix = load_data['label_l'] #假设文件中存有字符变量是matrix，例如matlab中save(load_fn, 'matrix');当然可以保存多个save(load_fn, 'matrix_x', 'matrix_y', ...);
    test_label1=np.array(load_matrix,dtype='float32')
    
#    test_label1=np.load(r'E:\ysdeeplearn\deepcode\deeptrain\何飞\yumu\train_label.npy')
    
    n=np.shape(test_label1)[1]
    
    test_label=np.zeros((n,2))
#        
#        for i in range(n):
#            if (test_label1[0,i]==1):
#                test_label[i,:]=[1 ,0,0,0]
#            elif (test_label1[0,i] == 2):
#                 test_label[i,:]=[0 ,1,0,0]
#            elif (test_label1[0,i] == 3):
#                 test_label[i,:]=[0 ,0,1,0]
#            elif (test_label1[0,i] == 4):
#                 test_label[i,:]=[0 ,0,0,1]
    for i in range(n):
        if test_label1[0,i]==1:
            test_label[i,:]=[1 ,0]
        elif (test_label1[0,i] > 1):
            test_label[i,:]=[0 ,1]
        
            
#    ff=np.where(np.isnan(train_data))[0]
#    dd=np.unique(ff)
#    train_data=np.delete(train_data, dd, axis=0)
#    test_label=np.delete(test_label, dd, axis=0)
    
#    idx=np.where(train_label==1)[1]
#        #idx=idx.reshape(1,58)
#    #    print(idx[-1])
#        
#    #    train_data=train_data*1024
###    train_data=np.transpose(train_data)
#    mu = np.mean(train_data,axis=0)
#    sigma = np.std(train_data,axis=0)
#########        train_data = preprocessing.normalize(train_data, norm='l2')
########    
#########        zui=np.max(train_data)
#########        
#    train_data=(train_data-mu)/sigma
#    train_data=np.transpose(train_data)
###    train_data=np.log(train_data)
#    train_data=np.transpose(train_data)
    mu = np.mean(train_data,axis=0)
    sigma = np.std(train_data,axis=0)
#######    
##########        train_data = preprocessing.normalize(train_data, norm='l2')
#########    
##########        zui=np.max(train_data)
##########        
    train_data=(train_data-mu)/sigma
#    train_data=np.transpose(train_data)
#        train_data=train_data/zui
    #
    #
#    test_data1=train_data[idx[0]:(idx[-1]+1),:]
#    test_label2=test_label[idx[0]:(idx[-1]+1),:]                     
    
#    train_data_copy=train_data
#    test_label_copy=test_label
    
#    train_data_shan=np.delete(train_data_copy, idx, axis=0)
#    test_label_shan=np.delete(test_label_copy, idx, axis=0)
#    
    l1=np.shape(train_data)[0]
    #l2=len(l1)
    
    idx1=rd.sample(range(l1),l1) 
#    
    train_data1=train_data[idx1,:]
    train_label1=test_label[idx1,:]
    return train_data1,train_label1

train_dataall,train_labelall=get()

jj=0

kf = KFold(n_splits=10,shuffle=True)
for train_index , test_index in kf.split(train_labelall):
#    l1=len(train_index)
#    idx1=rd.sample(range(l1),l1)
#    l2=len(test_index)
#    idx2=rd.sample(range(l2),l2) 
    
    train_data=train_dataall[train_index,:]
    train_label=train_labelall[train_index,:]
    test_data=train_dataall[test_index,:]
    test_label=train_labelall[test_index,:]
    jj=jj+1
#    print('train_index:%s , test_index: %s ' %(train_index,test_index))
#for i,j in kf:
#    print(i)
#    print(j)
#线性归一化方法
#from func_linear_normalization import func_linear_normalization
#from func_mean_normalization import func_mean_normalization

#for jj in range(138,149):
    s=[]
    s1=[]
    #load_test_label = r'E:\ysdeeplearn\deepcode\deeptrain\sp256\基于人的\12\test_label.mat'
    #load_data = sio.loadmat(load_test_label)
    #load_matrix = load_data['test_label'] #假设文件中存有字符变量是matrix，例如matlab中save(load_fn, 'matrix');当然可以保存多个save(load_fn, 'matrix_x', 'matrix_y', ...);
    #test_label1=np.array(load_matrix,dtype='float32')
    
    
    
#    def get():
#        load_train_label= r'E:\ysdeeplearn\deepcode\deeptrain\何飞\yumu\order_mfcc.mat'
#        load_data = sio.loadmat(load_train_label)
#        load_matrix = load_data['order_l'] #假设文件中存有字符变量是matrix，例如matlab中save(load_fn, 'matrix');当然可以保存多个save(load_fn, 'matrix_x', 'matrix_y', ...);
#        train_label=np.array(load_matrix,dtype='float32')
#        
#        
#        load_train_data=r'E:\ysdeeplearn\deepcode\deeptrain\何飞\yumu\data_mfcc.mat'
#        load_data = sio.loadmat(load_train_data)
#        load_matrix = load_data['data_l'] #假设文件中存有字符变量是matrix，例如matlab中save(load_fn, 'matrix');当然可以保存多个save(load_fn, 'matrix_x', 'matrix_y', ...);
#        train_data=np.array(load_matrix,dtype='float32')
#        
#        
#        
#        load_test_label = r'E:\ysdeeplearn\deepcode\deeptrain\何飞\yumu\label_mfcc.mat'
#        load_data = sio.loadmat(load_test_label)
#        load_matrix = load_data['label_l'] #假设文件中存有字符变量是matrix，例如matlab中save(load_fn, 'matrix');当然可以保存多个save(load_fn, 'matrix_x', 'matrix_y', ...);
#        test_label1=np.array(load_matrix,dtype='float32')
#        
#        n=np.shape(test_label1)[1]
#        
#        test_label=np.zeros((n,2))
##        
##        for i in range(n):
##            if (test_label1[0,i]==1):
##                test_label[i,:]=[1 ,0,0,0]
##            elif (test_label1[0,i] == 2):
##                 test_label[i,:]=[0 ,1,0,0]
##            elif (test_label1[0,i] == 3):
##                 test_label[i,:]=[0 ,0,1,0]
##            elif (test_label1[0,i] == 4):
##                 test_label[i,:]=[0 ,0,0,1]
#        for i in range(n):
#            if test_label1[0,i]==1:
#                test_label[i,:]=[1 ,0]
#            elif (test_label1[0,i] > 1):
#                test_label[i,:]=[0 ,1]
#            
#                
#        
#        idx=np.where(train_label==jj)[1]
##        #idx=idx.reshape(1,58)
##    #    print(idx[-1])
##        
##    #    train_data=train_data*1024
#        mu = np.mean(train_data,axis = 0)
#        sigma = np.std(train_data,axis = 0)
###        train_data = preprocessing.normalize(train_data, norm='l2')
##    
###        zui=np.max(train_data)
###        
#        train_data=(train_data-mu)/sigma
##        train_data=train_data/zui
#        #
#        #
#        test_data1=train_data[idx[0]:(idx[-1]+1),:]
#        test_label2=test_label[idx[0]:(idx[-1]+1),:]                     
#        
#        train_data_copy=train_data
#        test_label_copy=test_label
#        
#        train_data_shan=np.delete(train_data_copy, idx, axis=0)
#        test_label_shan=np.delete(test_label_copy, idx, axis=0)
#        
#        l1=np.shape(train_data_shan)[0]
#        #l2=len(l1)
#        
#        idx1=rd.sample(range(l1),l1) 
#        
#        train_data_shan1=train_data_shan[idx1,:]
#        train_label_shan=test_label_shan[idx1,:]
#        return train_data_shan1,train_label_shan,test_data1,test_label2
#    
#    train_data,train_label,test_data,test_label=get()
    with tf.Graph().as_default(): 
        steps_per_validate = 100
        steps_per_test = 100
        total_steps = 3000
        param_dict = {
                'lr' : 0.0001,
                'training_iters' : 3000,
                'batch_size' : 90, #choice_num,
                'n_inputs' : 129,#train_data_width,   # data input 
                'n_steps' :12,    # time steps
                'n_hidden_units' : 1024,   # neurons in hidden layer
                'n_classes' : 2,     # classes 
                'layer_num' : 3  #layers num
                }
    #    all_data_train_accuracy, test_accuracy = run_rnn_demo10(train_data, train_label, test_data, test_label, param_dict, i)
        lr = param_dict['lr']
        training_iters = param_dict['training_iters']
        #batch_size = param_dict['batch_size']
    
        n_inputs = param_dict['n_inputs']   # data input
        n_steps = param_dict['n_steps']    # time steps
        n_hidden_units = param_dict['n_hidden_units']   # neurons in hidden layer
        n_classes = param_dict['n_classes']     # classes 
    
        layer_num = param_dict['layer_num']  #layers num
    
    #    train_data_len = param_dict['train_data_len']
    #    train_data_width = param_dict['train_data_width']
    #
    #    choice_num_not = param_dict['choice_num_not']
    
        batch_size = tf.placeholder(tf.int32, [])
        keep_prob = tf.placeholder(tf.float32)
    
        _x = tf.placeholder(tf.float32, [None, 1548])
        y = tf.placeholder(tf.float32, [None, n_classes])
        x = tf.reshape(_x, [-1, n_steps, n_inputs])
        
        # tf Graph input
        #x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
        #y = tf.placeholder(tf.float32, [None, n_classes])
    
        # Define weights
        weights = {
            # (28, 128)
            'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
            # (128, 10)
            'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
        }
        biases = {
            # (128, )
            'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
            # (10, )
            'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
        }
    
        def RNN(X, weights, biases):
            # hidden layer for input to cell
            ########################################
    
            # transpose the inputs shape from
            # X ==> (128 batch * 28 steps, 28 inputs)
            X = tf.reshape(X, [-1, n_inputs])
    
            # into hidden
            # X_in = (128 batch * 28 steps, 128 hidden)
            X_in = tf.matmul(X, weights['in']) + biases['in']
            # X_in ==> (128 batch, 28 steps, 128 hidden)
            X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])
    
            # cell
            ##########################################
    
            # basic LSTM Cell.
            if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
                cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
            else:
                cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)
            # lstm cell is divided into two parts (c_state, h_state)
            #init_state = cell.zero_state(batch_size, dtype=tf.float32)
    
            #添加 dropout layer, 一般只设置 output_keep_prob
            cell = rnn.DropoutWrapper(cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
            #调用 MultiRNNCell 来实现多层 LSTM
            cell = rnn.MultiRNNCell([cell] * layer_num, state_is_tuple=True)
    
            # lstm cell is divided into two parts (c_state, h_state)
            init_state = cell.zero_state(batch_size, dtype=tf.float32)
    
            # You have 2 options for following step.
            # 1: tf.nn.rnn(cell, inputs);
            # 2: tf.nn.dynamic_rnn(cell, inputs).
            # If use option 1, you have to modified the shape of X_in, go and check out this:
            # https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py
            # In here, we go for option 2.
            # dynamic_rnn receive Tensor (batch, steps, inputs) or (steps, batch, inputs) as X_in.
            # Make sure the time_major is changed accordingly.
            outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)
    
            # hidden layer for output as the final results
            #############################################
            # results = tf.matmul(final_state[1], weights['out']) + biases['out']
    
            # # or
            # unpack to list [(batch, outputs)..] * steps
            # unpack to list [(batch, outputs)..] * steps
            if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
                outputs = tf.unpack(tf.transpose(outputs, [1, 0, 2]))    # states is the last outputs
            else:
                outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))
#            outputs1=np.mean(outputs,axis=0)
#            outputs1=tf.reduce_mean(outputs, 0)
#            results = tf.matmul(outputs[-1], weights['out']) + biases['out']  
#            results = tf.matmul(outputs1, weights['out']) + biases['out'] # shape = (128, 10)
    
            return outputs
        outputs = RNN(x, weights, biases)
    
        on_train = tf.placeholder(tf.bool)
        
        #load_train_label= r'E:\ysdeeplearn\deepcode\deeptrain\sp256\基于人的\12\train_label.mat'
        #load_data = sio.loadmat(load_train_label)
        #load_matrix = load_data['train_label'] #假设文件中存有字符变量是matrix，例如matlab中save(load_fn, 'matrix');当然可以保存多个save(load_fn, 'matrix_x', 'matrix_y', ...);
        #train_label1=np.array(load_matrix,dtype='float32')
        #
        #load_train_data= r'E:\ysdeeplearn\deepcode\deeptrain\sp256\基于人的\12\train_data.mat'
        #load_data = sio.loadmat(load_train_data)
        #load_matrix = load_data['train_data'] #假设文件中存有字符变量是matrix，例如matlab中save(load_fn, 'matrix');当然可以保存多个save(load_fn, 'matrix_x', 'matrix_y', ...);
        #train_data=np.array(load_matrix,dtype='float32')
        #
        #load_test_data= r'E:\ysdeeplearn\deepcode\deeptrain\sp256\基于人的\12\test_data.mat'
        #load_data = sio.loadmat(load_test_data)
        #load_matrix = load_data['test_data'] #假设文件中存有字符变量是matrix，例如matlab中save(load_fn, 'matrix');当然可以保存多个save(load_fn, 'matrix_x', 'matrix_y', ...);
        #test_data=np.array(load_matrix,dtype='float32')
        # number 1 to 10 data
        
        #f1=len(train_label1)
        #train_label=np.zeros((f1,2))
        #f2=len(test_label1)
        #test_label=np.zeros((f2,2))
        #for i in range(f1):
        #    if train_label1[i,0]==1:
        #        train_label[i,:]=[1,0]
        #    else:
        #        train_label[i,:]=[0,1]
        #
        #for j in range(f2):
        #    if test_label1[j,0]==1:
        #        test_label[j,:]=[1,0]
        #    else:
        #        test_label[j,:]=[0,1]
        
        
        
        
#        def compute_accuracy(v_xs, v_ys,norm):
#            global prediction
#            global s
#            global s1
#            global on_train
#            y_pre = sess.run(prediction, feed_dict={xs: v_xs,on_train: norm , keep_prob: 1,pkeep_conv:1})
#            correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
#            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#            result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, on_train: norm ,keep_prob: 1,pkeep_conv:1})
#            s=y_pre
#            t=result
#            s1.append(t)
#            return result
        
        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)
        
        def bias_variable(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)
        
        def conv2d(x, W):
            # stride [1, x_movement, y_movement, 1]
            # Must have strides[0] = strides[3] = 1
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
        
        def max_pool_2x2(x):
            # stride [1, x_movement, y_movement, 1]
            return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        
        # define placeholder for inputs to network
#        _x = tf.placeholder(tf.float32, [None, 1548])   # 28x28
#        ys = tf.placeholder(tf.float32, [None, 4])
        
        
        
        
        
        
        #
        #fc_mean, fc_var = tf.nn.moments(
        #            xs1,
        #            axes=[0],
        #        )
        #scale = tf.Variable(tf.ones([480]))
        #shift = tf.Variable(tf.zeros([480]))
        #epsilon = 0.001
        #        # apply moving average for mean and var when train on batch
        #ema = tf.train.ExponentialMovingAverage(decay=0.5)
        #def mean_var_with_update():
        #    ema_apply_op = ema.apply([fc_mean, fc_var])
        #    with tf.control_dependencies([ema_apply_op]):
        #        return tf.identity(fc_mean), tf.identity(fc_var)
        #mean, var = mean_var_with_update()
        #xs = tf.nn.batch_normalization(xs1, mean, var, shift, scale, epsilon)
        
        
        
        
        
        
        
        
        
        
        
        
        pkeep_conv = tf.placeholder(tf.float32)
        
        
#        keep_prob = tf.placeholder(tf.float32)
        x_image = tf.reshape(_x, [-1, 12, 129, 1])
        #x_image=tf.transpose(x_image)
        #x_image=tf.image.resize_images(x_image1,[16,240],method=0)
        # print(x_image.shape)  # [n_samples, 28,28,1]
        
        ## conv1 layer ##
        W_conv1 = weight_variable([5,5, 1,32]) # patch 5x5, in size 1, out size 32
        b_conv1 = bias_variable([32])
        h6=conv2d(x_image, W_conv1) + b_conv1
        
#        fc_mean4, fc_var4 = tf.nn.moments(
#                        h6,
#                        axes=[0,1,2],   # the dimension you wanna normalize, here [0] for batch
#                                    # for image, you wanna do [0, 1, 2] for [batch, height, width] but not channel
#                    )
#        scale4 = tf.Variable(tf.ones([32]))
#        shift4 = tf.Variable(tf.zeros([32]))
#        epsilon4 = 0.001
#        
#                    # apply moving average for mean and var when train on batch
#        ema = tf.train.ExponentialMovingAverage(decay=0.90)
#        def mean_var_with_update():
#            ema_apply_op = ema.apply([fc_mean4, fc_var4])
#            with tf.control_dependencies([ema_apply_op]):
#                return tf.identity(fc_mean4), tf.identity(fc_var4)
#        
#        mean4, var4 = tf.cond(on_train,    # on_train 的值是 True/False
#                            mean_var_with_update,   # 如果是 True, 更新 mean/var
#                            lambda: (               # 如果是 False, 返回之前 fc_mean/fc_var 的Moving Average
#                                ema.average(fc_mean4), 
#                                ema.average(fc_var4)
#                                )    
#                            )
#        #mean, var = mean_var_with_update()
#        hcon = tf.nn.batch_normalization(h6, mean4, var4, shift4, scale4, epsilon4)
        
        
        h_conv1r = tf.nn.relu(h6)
        h_conv1 = tf.nn.dropout(h_conv1r, pkeep_conv) # output size 28x28x32
        h_pool1 = max_pool_2x2(h_conv1) 
        #h_pool1 = tf.nn.dropout(h_pool1r, pkeep_conv)                                        # output size 14x14x32
        
        # conv2 layer ##
        W_conv2 = weight_variable([5,5, 32, 64]) # patch 5x5, in size 32, out size 64
        b_conv2 = bias_variable([64])
        h7=conv2d(h_pool1, W_conv2) + b_conv2
#        
##        fc_mean5, fc_var5 = tf.nn.moments(
##                        h7,
##                        axes=[0,1,2],   # the dimension you wanna normalize, here [0] for batch
##                                    # for image, you wanna do [0, 1, 2] for [batch, height, width] but not channel
##                    )
##        scale5 = tf.Variable(tf.ones([64]))
##        shift5 = tf.Variable(tf.zeros([64]))
##        epsilon5 = 0.001
##        
##                    # apply moving average for mean and var when train on batch
##        ema = tf.train.ExponentialMovingAverage(decay=0.90)
##        def mean_var_with_update():
##            ema_apply_op = ema.apply([fc_mean5, fc_var5])
##            with tf.control_dependencies([ema_apply_op]):
##                return tf.identity(fc_mean5), tf.identity(fc_var5)
##        
##        mean5, var5 = tf.cond(on_train,    # on_train 的值是 True/False
##                            mean_var_with_update,   # 如果是 True, 更新 mean/var
##                            lambda: (               # 如果是 False, 返回之前 fc_mean/fc_var 的Moving Average
##                                ema.average(fc_mean5), 
##                                ema.average(fc_var5)
##                                )    
##                            )
##        #mean, var = mean_var_with_update()
##        hcon1 = tf.nn.batch_normalization(h7, mean5, var5, shift5, scale5, epsilon5)
#        
        h_conv2r = tf.nn.relu(h7)
        h_conv2 = tf.nn.dropout(h_conv2r, pkeep_conv)  # output size 14x14x64
        h_pool2 = max_pool_2x2(h_conv2)                                         # output size 7x7x64
        #h_pool2 = tf.nn.dropout(h_pool2r, pkeep_conv)
        
        
        ## fc1 layer ##
        W_fc1 = weight_variable([3*33*64+1024, 1024])
        b_fc1 = bias_variable([1024])
        # [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
        h_pool2_flat = tf.reshape(h_pool2, [-1, 3*33*64])
        
        h_pool2_flat=tf.concat( [h_pool2_flat, outputs[-1]],1)
        
        h1=tf.matmul(h_pool2_flat, W_fc1) + b_fc1
        
#        h1=tf.concat( [h1, outputs[-1]])
        
        fc_mean, fc_var = tf.nn.moments(
                        h1,
                        axes=[0,],   # the dimension you wanna normalize, here [0] for batch
                                    # for image, you wanna do [0, 1, 2] for [batch, height, width] but not channel
                    )
        scale = tf.Variable(tf.ones([1024]))
        shift = tf.Variable(tf.zeros([1024]))
        epsilon = 0.001
        
                    # apply moving average for mean and var when train on batch
        ema = tf.train.ExponentialMovingAverage(decay=0.9)
        def mean_var_with_update():
            ema_apply_op = ema.apply([fc_mean, fc_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(fc_mean), tf.identity(fc_var)
        
        mean, var = tf.cond(on_train,    # on_train 的值是 True/False
                            mean_var_with_update,   # 如果是 True, 更新 mean/var
                            lambda: (               # 如果是 False, 返回之前 fc_mean/fc_var 的Moving Average
                                ema.average(fc_mean), 
                                ema.average(fc_var)
                                )    
                            )
        #mean, var = mean_var_with_update()
        Wx_plus_b = tf.nn.batch_normalization(h1, mean, var, shift, scale, epsilon)
        
        h_fc1 = tf.nn.relu(Wx_plus_b)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        
        
        #mean, var = tf.cond(on_train,    # on_train 的值是 True/False
        #                    mean_var_with_update,   # 如果是 True, 更新 mean/var
        #                    lambda: (               # 如果是 False, 返回之前 fc_mean/fc_var 的Moving Average
        #                        ema.average(fc_mean), 
        #                        ema.average(fc_var)
        #                        )    
        #                    )
        
        #3 layer
        
        
        
        W_fc3 = weight_variable([1024, 1024])
        b_fc3 = bias_variable([1024])
        h3=tf.matmul(h_fc1_drop, W_fc3) + b_fc3
        fc_mean2, fc_var2 = tf.nn.moments(
                        h3,
                        axes=[0],   # the dimension you wanna normalize, here [0] for batch
                                    # for image, you wanna do [0, 1, 2] for [batch, height, width] but not channel
                    )
        scale2 = tf.Variable(tf.ones([1024]))
        shift2 = tf.Variable(tf.zeros([1024]))
        epsilon2 = 0.001
        
                    # apply moving average for mean and var when train on batch
        ema = tf.train.ExponentialMovingAverage(decay=0.9)
        def mean_var_with_update():
            ema_apply_op = ema.apply([fc_mean2, fc_var2])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(fc_mean2), tf.identity(fc_var2)
        #mean2, var2 = mean_var_with_update()
        mean2, var2 = tf.cond(on_train,    # on_train 的值是 True/False
                            mean_var_with_update,   # 如果是 True, 更新 mean/var
                            lambda: (               # 如果是 False, 返回之前 fc_mean/fc_var 的Moving Average
                                ema.average(fc_mean2), 
                                ema.average(fc_var2)
                                )    
                            )
        Wx_plus_b2 = tf.nn.batch_normalization(h3, mean2, var2, shift2, scale2, epsilon2)
        h_fc2 = tf.nn.relu(Wx_plus_b2)
        h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)
        
        
        
        
        
        
        
        
        ## fc2 layer ##
        W_fc2 = weight_variable([1024, 2])
        b_fc2 = bias_variable([2])
        h2=tf.matmul(h_fc2_drop, W_fc2) + b_fc2
        #fc_mean1, fc_var1 = tf.nn.moments(
        #                h2,
        #                axes=[0],   # the dimension you wanna normalize, here [0] for batch
        #                            # for image, you wanna do [0, 1, 2] for [batch, height, width] but not channel
        #            )
        #scale1 = tf.Variable(tf.ones([4]))
        #shift1 = tf.Variable(tf.zeros([4]))
        #epsilon1 = 0.001
        #
        #            # apply moving average for mean and var when train on batch
        #ema = tf.train.ExponentialMovingAverage(decay=0.5)
        #def mean_var_with_update():
        #    ema_apply_op = ema.apply([fc_mean1, fc_var1])
        #    with tf.control_dependencies([ema_apply_op]):
        #        return tf.identity(fc_mean1), tf.identity(fc_var1)
        ##mean1, var1 = mean_var_with_update()
        #mean1, var1 = tf.cond(on_train,    # on_train 的值是 True/False
        #                    mean_var_with_update,   # 如果是 True, 更新 mean/var
        #                    lambda: (               # 如果是 False, 返回之前 fc_mean/fc_var 的Moving Average
        #                        ema.average(fc_mean1), 
        #                        ema.average(fc_var1)
        #                        )    
        #                    )
        #Wx_plus_b1 = tf.nn.batch_normalization(h2, mean1, var1, shift1, scale1, epsilon1)
        
        
        pred = tf.nn.softmax(h2)
        
        
        # the error between prediction and real data
        #
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=pred)
        cross_entropy = tf.reduce_mean(cross_entropy)
        train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)
        
        
#        pred,outputs = RNN(x, weights, biases)
#        pred = tf.nn.softmax(pred)
#        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
#        train_op = tf.train.AdamOptimizer(lr).minimize(cost)
    
        
    
    
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        def get_batch_data():
            [label, images] = [train_label,train_data]
            images = tf.cast(images, tf.float32)
            label = tf.cast(label, tf.float32)
            input_queue = tf.train.slice_input_producer([images, label], shuffle=False)
            image_batch, label_batch = tf.train.batch(input_queue, batch_size=256, num_threads=1, capacity=2048)
            return image_batch, label_batch
        image_batch, label_batch = get_batch_data()
        # Train
        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess, coord)
            sess.run(tf.global_variables_initializer())
            for step in range(total_steps + 1):
                image_batch_v, label_batch_v = sess.run([image_batch, label_batch])
                sess.run(train_step, feed_dict={_x: image_batch_v, y: label_batch_v,on_train: True ,keep_prob: 0.5, batch_size: image_batch_v.shape[0],pkeep_conv:1})
                # Train Accuracy
                
                if step % steps_per_validate == 0:
                    t1= sess.run(accuracy, feed_dict={_x:image_batch_v, y: label_batch_v, keep_prob: 1,on_train: False,batch_size: image_batch_v.shape[0],pkeep_conv:1})
                    print('Train', step,t1)
                    s1.append(t1)
                # Test Accuracy
                if step % steps_per_test == 0:
                    t= sess.run(accuracy, feed_dict={_x: test_data, y: test_label, keep_prob: 1,on_train: False, batch_size: test_data.shape[0],pkeep_conv:1})
        #            test_x, test_y = mnist.test.images, mnist.test.labels
                    print('Test', step,t)
                    s1.append(t)
                    s=sess.run(pred, feed_dict={_x: test_data, y: test_label, keep_prob: 1,on_train: False, batch_size: test_data.shape[0],pkeep_conv:1})
        new1=os.path.join(path,str(jj))
        new2=new1+p
        new3=new1+p1
        np.save(new2,s)
        np.save(new3,s1)
        if jj==1:
            break