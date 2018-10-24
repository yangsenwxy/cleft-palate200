# -*- coding: utf-8 -*-
"""
Created on Tue May 15 10:50:13 2018

@author: Administrator
"""

import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops.rnn_cell_impl import DropoutWrapper
import scipy.io as sio 
import numpy as np
import random as rd
import os

# Get Mnist Datapath
path="E:\ysdeeplearn\结果2\\"
p='s.npy'
p1='s1.npy'
#mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
for jj in range(50,60):
    s=np.array([])
    s1=[]
    #load_test_label = r'E:\ysdeeplearn\deepcode\deeptrain\sp256\基于人的\12\test_label.mat'
    #load_data = sio.loadmat(load_test_label)
    #load_matrix = load_data['test_label'] #假设文件中存有字符变量是matrix，例如matlab中save(load_fn, 'matrix');当然可以保存多个save(load_fn, 'matrix_x', 'matrix_y', ...);
    #test_label1=np.array(load_matrix,dtype='float32')
    
    
    
    def get():
        load_train_label= r'E:\ysdeeplearn\deepcode\deeptrain\何飞\yumu\order_LSTM20.mat'
        load_data = sio.loadmat(load_train_label)
        load_matrix = load_data['order_l'] #假设文件中存有字符变量是matrix，例如matlab中save(load_fn, 'matrix');当然可以保存多个save(load_fn, 'matrix_x', 'matrix_y', ...);
        train_label=np.array(load_matrix,dtype='float32')
        
        
        load_train_data= r'E:\ysdeeplearn\deepcode\deeptrain\何飞\yumu\data_LSTM20.mat'
        load_data = sio.loadmat(load_train_data)
        load_matrix = load_data['data_l'] #假设文件中存有字符变量是matrix，例如matlab中save(load_fn, 'matrix');当然可以保存多个save(load_fn, 'matrix_x', 'matrix_y', ...);
        train_data=np.array(load_matrix,dtype='float32')
        
        
        
        load_test_label = r'E:\ysdeeplearn\deepcode\deeptrain\何飞\yumu\label_LSTM20.mat'
        load_data = sio.loadmat(load_test_label)
        load_matrix = load_data['label_l'] #假设文件中存有字符变量是matrix，例如matlab中save(load_fn, 'matrix');当然可以保存多个save(load_fn, 'matrix_x', 'matrix_y', ...);
        test_label1=np.array(load_matrix,dtype='float32')
        
        n=np.shape(test_label1)[1]
        
        test_label=np.zeros((n,4))
        
        for i in range(n):
            if (test_label1[0,i]==1):
                test_label[i,:]=[1 ,0,0,0]
            elif (test_label1[0,i] == 2):
                 test_label[i,:]=[0 ,1,0,0]
            elif (test_label1[0,i] == 3):
                 test_label[i,:]=[0 ,0,1,0]
            elif (test_label1[0,i] == 4):
                 test_label[i,:]=[0 ,0,0,1]
            
                
        
        idx=np.where(train_label==jj)[1]
        #idx=idx.reshape(1,58)
    #    print(idx[-1])
        
    #    train_data=train_data*1024
        mu = np.mean(train_data,axis = 0)
        sigma = np.std(train_data,axis = 0)
    
    #    zui=np.max(train_data)
        
        train_data=(train_data-mu)/sigma
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
    # Variable
        learning_rate = 1e-3
        num_units = 256
        num_layer = 3
        input_size = 280
        time_step = 10
        total_steps = 3000
        category_num = 4
        steps_per_validate = 100
        steps_per_test = 100
        batch_size = tf.placeholder(tf.int32, [])
        keep_prob = tf.placeholder(tf.float32, [])
        
        
        # Get RNN Cell
        def cell(num_units):
            cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=num_units,state_is_tuple=True)
            return DropoutWrapper(cell, output_keep_prob=keep_prob)
        
        
        # Initial
        x = tf.placeholder(tf.float32, [None, 2800])
        y_label = tf.placeholder(tf.float32, [None, 4])
        x_shape = tf.reshape(x, [-1, time_step, input_size])
        x_shape = tf.transpose(x_shape, [1, 0, 2])
        x_shape = tf.reshape(x_shape, [-1, input_size ])
        x_shape = tf.split(x_shape, time_step)
        # RNN Layers
        cells_fw = tf.nn.rnn_cell.MultiRNNCell([cell(num_units) for j in range(num_layer)])
        cells_bw = tf.nn.rnn_cell.MultiRNNCell([cell(num_units) for j in range(num_layer)])
        outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(cells_fw, cells_bw , x_shape, dtype = tf.float32)
#        h0 = cells.zero_state(batch_size, dtype=tf.float32)
#        output, hs = tf.nn.dynamic_rnn(cells, inputs=x_shape, initial_state=h0,time_major=False)
#        output = output[:, -1, :]
        # Or h = hs[-1].h
        
        # Output Layer
#        w = tf.Variable(tf.truncated_normal([num_units, category_num], stddev=0.1), dtype=tf.float32)
#        b = tf.Variable(tf.constant(0.1, shape=[category_num]), dtype=tf.float32)
        W = tf.Variable(tf.random_normal([2 * num_units, category_num]))   #参数共享力度比cnn还大
        b = tf.Variable(tf.random_normal([category_num]))
        y = tf.nn.softmax(tf.matmul(outputs[-1], W) + b)
        
        # Loss
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_label, logits=y)
        train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
        
        # Prediction
        correction_prediction = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_label, axis=1))
        accuracy = tf.reduce_mean(tf.cast(correction_prediction, tf.float32))
        def get_batch_data():
            [label, images] = [train_label,train_data]
            images = tf.cast(images, tf.float32)
            label = tf.cast(label, tf.float32)
            input_queue = tf.train.slice_input_producer([images, label], shuffle=False)
            image_batch, label_batch = tf.train.batch(input_queue, batch_size=1024, num_threads=1, capacity=2048)
            return image_batch, label_batch
        image_batch, label_batch = get_batch_data()
        # Train
        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess, coord)
            sess.run(tf.global_variables_initializer())
            for step in range(total_steps + 1):
                image_batch_v, label_batch_v = sess.run([image_batch, label_batch])
                sess.run(train, feed_dict={x: image_batch_v, y_label: label_batch_v, keep_prob: 0.4, batch_size: image_batch_v.shape[0]})
                # Train Accuracy
                if step % steps_per_validate == 0:
                    t1= sess.run(accuracy, feed_dict={x:image_batch_v, y_label: label_batch_v, keep_prob: 1,batch_size: image_batch_v.shape[0]})
                    print('Train', step,t1)
                    s1.append(t1)
                # Test Accuracy
                if step % steps_per_test == 0:
                    t= sess.run(accuracy, feed_dict={x: test_data, y_label: test_label, keep_prob: 1, batch_size: test_data.shape[0]})
        #            test_x, test_y = mnist.test.images, mnist.test.labels
                    print('Test', step,t)
                    s1.append(t)
                    s=sess.run(y, feed_dict={x: test_data, y_label: test_label, keep_prob: 1, batch_size: test_data.shape[0]})
        new1=os.path.join(path,str(jj))
        new2=new1+p
        new3=new1+p1
        np.save(new2,s)
        np.save(new3,s1)
        
                    