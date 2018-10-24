# -*- coding: utf-8 -*-
"""
Created on Sat May 26 14:26:22 2018

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
#from random_choice_list import random_choice_list
path="E:\ysdeeplearn\wuyong\\"
p='s.npy'
p1='s1.npy'



def get():
#    load_train_label= r'D:\ys\外星人\yumu\order_vocal_area.mat'
#    load_data = sio.loadmat(load_train_label)
#    load_matrix = load_data['order_l'] #假设文件中存有字符变量是matrix，例如matlab中save(load_fn, 'matrix');当然可以保存多个save(load_fn, 'matrix_x', 'matrix_y', ...);
#    train_label=np.array(load_matrix,dtype='float32')
    
    
    load_train_data=r'E:\ysdeeplearn\deepcode\deeptrain\何飞\yumu\data_cepstrum.mat'
    load_data = sio.loadmat(load_train_data)
    load_matrix = load_data['data_l'] #假设文件中存有字符变量是matrix，例如matlab中save(load_fn, 'matrix');当然可以保存多个save(load_fn, 'matrix_x', 'matrix_y', ...);
    train_data=np.array(load_matrix,dtype='float32')
#    train_data=np.load(r'D:\ys\外星人\yumu\train_cepstrum.npy')
    
    
    
    load_test_label = r'E:\ysdeeplearn\deepcode\deeptrain\何飞\yumu\label_cepstrum.mat'
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
        
            
    
#    idx=np.where(train_label==1)[1]
#        #idx=idx.reshape(1,58)
#    #    print(idx[-1])
#        
#    #    train_data=train_data*1024
##    train_data=np.transpose(train_data)
#    mu = np.mean(train_data,axis=0)
#    sigma = np.std(train_data,axis=0)
########        train_data = preprocessing.normalize(train_data, norm='l2')
#######    
########        zui=np.max(train_data)
########        
#    train_data=(train_data-mu)/sigma
#    train_data=np.transpose(train_data)
#    train_data=np.log(train_data)
###    train_data=np.transpose(train_data)
    mu = np.mean(train_data,axis=0)
    sigma = np.std(train_data,axis=0)
########    
###########        train_data = preprocessing.normalize(train_data, norm='l2')
##########    
###########        zui=np.max(train_data)
###########        
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
                'lr' : 0.00001,
                'training_iters' : 3000,
                'batch_size' : 90, #choice_num,
                'n_inputs' : 441,#train_data_width,   # data input 
                'n_steps' :12,    # time steps
                'n_hidden_units' : 2048,   # neurons in hidden layer
                'n_classes' : 2,     # classes 
                'layer_num' : 3 #layers num
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
    
        _x = tf.placeholder(tf.float32, [None, 5292])
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
                cell = tf.nn.rnn_cell.LSTMCell(n_hidden_units,use_peepholes=True, forget_bias=0.5, state_is_tuple=True)
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
            results = tf.matmul(outputs[-1], weights['out']) + biases['out']  
#            results = tf.matmul(outputs1, weights['out']) + biases['out'] # shape = (128, 10)
    
            return results,outputs
    
    
        pred,outputs = RNN(x, weights, biases)
        pred = tf.nn.softmax(pred)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
        train_op = tf.train.AdamOptimizer(lr).minimize(cost)
    
        
    
    
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        def get_batch_data():
            [label, images] = [train_label,train_data]
            images = tf.cast(images, tf.float32)
            label = tf.cast(label, tf.float32)
            input_queue = tf.train.slice_input_producer([images, label], shuffle=False)
            image_batch, label_batch = tf.train.batch(input_queue, batch_size=512, num_threads=1, capacity=2048)
            return image_batch, label_batch
        image_batch, label_batch = get_batch_data()
        # Train
        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess, coord)
            sess.run(tf.global_variables_initializer())
            for step in range(total_steps + 1):
                image_batch_v, label_batch_v = sess.run([image_batch, label_batch])
                _,ll=sess.run([train_op,outputs], feed_dict={_x: image_batch_v, y: label_batch_v, keep_prob: 0.5, batch_size: image_batch_v.shape[0]})
                # Train Accuracy
                
                if step % steps_per_validate == 0:
                    t1= sess.run(accuracy, feed_dict={_x:image_batch_v, y: label_batch_v, keep_prob: 1,batch_size: image_batch_v.shape[0]})
                    print('Train', step,t1)
                    s1.append(t1)
                # Test Accuracy
                if step % steps_per_test == 0:
                    t= sess.run(accuracy, feed_dict={_x: test_data, y: test_label, keep_prob: 1, batch_size: test_data.shape[0]})
        #            test_x, test_y = mnist.test.images, mnist.test.labels
                    print('Test', step,t)
                    s1.append(t)
                    s=sess.run(pred, feed_dict={_x: test_data, y: test_label, keep_prob: 1, batch_size: test_data.shape[0]})
        new1=os.path.join(path,str(jj))
        new2=new1+p
        new3=new1+p1
        np.save(new2,s)
        np.save(new3,s1)