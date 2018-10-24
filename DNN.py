# -*- coding: utf-8 -*-
"""
Created on Thu May 31 19:29:44 2018

@author: 18384
"""

from tensorflow.contrib import rnn
#from sklearn import preprocessing
#from run_rnn_demo10 import run_rnn_demo10
#from func_write_data_to_mat import func_write_data_to_mat

#from read_mat_to_list import read_mat_to_list
#from list_extend_to_list import list_extend_to_list
#from list_length import list_length
import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops.rnn_cell_impl import DropoutWrapper
import scipy.io as sio 
import numpy as np
import random as rd
import os

def add_layer(inputs, in_size, out_size,on_train ,keep_prob,activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    fc_mean, fc_var = tf.nn.moments(
                Wx_plus_b,
                axes=[0],   # the dimension you wanna normalize, here [0] for batch
                            # for image, you wanna do [0, 1, 2] for [batch, height, width] but not channel
            )
    scale = tf.Variable(tf.ones([out_size]))
    shift = tf.Variable(tf.zeros([out_size]))
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
    Wx_plus_b1 = tf.nn.batch_normalization(Wx_plus_b, mean, var, shift, scale, epsilon)
    if activation_function is None:
        outputs = Wx_plus_b1
    else:
        outputs = activation_function(Wx_plus_b1)
        outputs1 = tf.nn.dropout(outputs, keep_prob)
    return outputs1

#from random_choice_list import random_choice_list
path="E:\ysdeeplearn\结果DNN2\\"
p='s.npy'
p1='s1.npy'
#线性归一化方法
#from func_linear_normalization import func_linear_normalization
#from func_mean_normalization import func_mean_normalization

for jj in range(22,23):
    s=[]
    s1=[]
    #load_test_label = r'E:\ysdeeplearn\deepcode\deeptrain\sp256\基于人的\12\test_label.mat'
    #load_data = sio.loadmat(load_test_label)
    #load_matrix = load_data['test_label'] #假设文件中存有字符变量是matrix，例如matlab中save(load_fn, 'matrix');当然可以保存多个save(load_fn, 'matrix_x', 'matrix_y', ...);
    #test_label1=np.array(load_matrix,dtype='float32')
    
    
    
    def get():
        load_train_label= r'E:\ysdeeplearn\deepcode\deeptrain\何飞\yumu\order_yumufs.mat'
        load_data = sio.loadmat(load_train_label)
        load_matrix = load_data['order_l'] #假设文件中存有字符变量是matrix，例如matlab中save(load_fn, 'matrix');当然可以保存多个save(load_fn, 'matrix_x', 'matrix_y', ...);
        train_label=np.array(load_matrix,dtype='float32')
        
        
        load_train_data=r'E:\ysdeeplearn\deepcode\deeptrain\何飞\yumu\data_yumufs.mat'
        load_data = sio.loadmat(load_train_data)
        load_matrix = load_data['data_l'] #假设文件中存有字符变量是matrix，例如matlab中save(load_fn, 'matrix');当然可以保存多个save(load_fn, 'matrix_x', 'matrix_y', ...);
        train_data=np.array(load_matrix,dtype='float32')
        
        
        
        load_test_label = r'E:\ysdeeplearn\deepcode\deeptrain\何飞\yumu\label_yumufs.mat'
        load_data = sio.loadmat(load_test_label)
        load_matrix = load_data['label_l'] #假设文件中存有字符变量是matrix，例如matlab中save(load_fn, 'matrix');当然可以保存多个save(load_fn, 'matrix_x', 'matrix_y', ...);
        test_label1=np.array(load_matrix,dtype='float32')
        
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
            
                
        
        idx=np.where(train_label==jj)[1]
#        #idx=idx.reshape(1,58)
#    #    print(idx[-1])
#        
#    #    train_data=train_data*1024
        mu = np.mean(train_data,axis = 0)
        sigma = np.std(train_data,axis = 0)
##        train_data = preprocessing.normalize(train_data, norm='l2')
#    
##        zui=np.max(train_data)
##        
        train_data=(train_data-mu)/sigma
#        train_data=train_data/zui
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
    with tf.Graph().as_default(): 
         keep_prob = tf.placeholder(tf.float32)
         on_train = tf.placeholder(tf.bool)
         x = tf.placeholder(tf.float32, [None, 6144])
         W = tf.Variable(tf.zeros([6144, 2]))
         b = tf.Variable(tf.zeros([2]))
        
        # add hidden layer
         l1 = add_layer(x, 6144, 1024,on_train,keep_prob, activation_function=tf.nn.relu)
         l2 = add_layer(l1, 1024, 2048,on_train,keep_prob, activation_function=tf.nn.relu)
         l3 = add_layer(l2, 2048, 512,on_train,keep_prob, activation_function=tf.nn.relu)
        # add output layer
         y = add_layer(l3, 512, 2,on_train,keep_prob, activation_function=tf.nn.softmax)
        
        
        #y = tf.matmul(x, W) + b
        
          # Define loss and optimizer
         y_ = tf.placeholder(tf.float32, [None, 2])
        
          # The raw formulation of cross-entropy,
          #
          #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
          #                                 reduction_indices=[1]))
          #
          # can be numerically unstable.
          #
          # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
          # outputs of 'y', and then average across the batch.
         cross_entropy = tf.reduce_mean(
               tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
         train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
         correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
         accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
         def get_batch_data():
             [label, images] = [train_label,train_data]
             images = tf.cast(images, tf.float32)
             label = tf.cast(label, tf.float32)
             input_queue = tf.train.slice_input_producer([images, label], shuffle=False)
             image_batch, label_batch = tf.train.batch(input_queue, batch_size=1024, num_threads=1, capacity=2048)
             return image_batch, label_batch
         image_batch, label_batch = get_batch_data()
         with tf.Session() as sess:
             coord = tf.train.Coordinator()
             threads = tf.train.start_queue_runners(sess, coord)
             sess.run(tf.global_variables_initializer())
             for step in range(1000):
                 image_batch_v, label_batch_v = sess.run([image_batch, label_batch])
                 sess.run(train_step, feed_dict={x: image_batch_v, y_: label_batch_v,on_train: True, keep_prob: 0.5})
                # Train Accuracy
                 if step % 100 == 0:
                    t1= sess.run(accuracy, feed_dict={x:image_batch_v, y_: label_batch_v,on_train: False, keep_prob: 1})
                    print('Train', step,t1)
                    s1.append(t1)
                # Test Accuracy
                 if step % 100 == 0:
                    t= sess.run(accuracy, feed_dict={x: test_data, y_: test_label,on_train: False, keep_prob: 1})
        #            test_x, test_y = mnist.test.images, mnist.test.labels
                    print('Test', step,t)
                    s1.append(t)
                    s=sess.run(y, feed_dict={x: test_data, y_: test_label, on_train: False,keep_prob: 1})
         new1=os.path.join(path,str(jj))
         new2=new1+p
         new3=new1+p1
         np.save(new2,s)
         np.save(new3,s1)
        
#         sess = tf.InteractiveSession()
#         tf.global_variables_initializer().run()
            