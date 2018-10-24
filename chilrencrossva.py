# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 16:21:33 2017

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
path="E:\ysdeeplearn\交叉验证cnn4类\\"
p='s.npy'
p1='s1.npy'



def get():
    load_train_label= r'E:\ysdeeplearn\deepcode\deeptrain\何飞\yumu\order_yumufs.mat'
    load_data = sio.loadmat(load_train_label)
    load_matrix = load_data['order_l'] #假设文件中存有字符变量是matrix，例如matlab中save(load_fn, 'matrix');当然可以保存多个save(load_fn, 'matrix_x', 'matrix_y', ...);
    train_label=np.array(load_matrix,dtype='float32')
    
    
    load_train_data=r'E:\ysdeeplearn\deepcode\deeptrain\何飞\yumu\四类所有\四类\data_yumufs.mat'
    load_data = sio.loadmat(load_train_data)
    load_matrix = load_data['data_l'] #假设文件中存有字符变量是matrix，例如matlab中save(load_fn, 'matrix');当然可以保存多个save(load_fn, 'matrix_x', 'matrix_y', ...);
    train_data=np.array(load_matrix,dtype='float32')
    
    
    
    load_test_label = r'E:\ysdeeplearn\deepcode\deeptrain\何飞\yumu\四类所有\四类\label_yumufs.mat'
    load_data = sio.loadmat(load_test_label)
    load_matrix = load_data['label_l'] #假设文件中存有字符变量是matrix，例如matlab中save(load_fn, 'matrix');当然可以保存多个save(load_fn, 'matrix_x', 'matrix_y', ...);
    test_label1=np.array(load_matrix,dtype='float32')
    
    n=np.shape(test_label1)[1]
    
    test_label=np.zeros((n,4))
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
            test_label[i,:]=[1 ,0,0,0]
        if (test_label1[0,i] ==2):
            test_label[i,:]=[0 ,1,0,0]
        if (test_label1[0,i] ==3):
            test_label[i,:]=[0,0 ,1,0]
        if (test_label1[0,i] == 4):
            test_label[i,:]=[0,0, 0,1]
        
            
#    train_data=np.log(train_data)
#    idx=np.where(train_label==1)[1]
#        #idx=idx.reshape(1,58)
#    #    print(idx[-1])
#        
#    #    train_data=train_data*1024
#    train_data=np.transpose(train_data)
    train_data=np.log(train_data)
    mu = np.mean(train_data,axis=0)
    sigma = np.std(train_data,axis=0)
#    
####        train_data = preprocessing.normalize(train_data, norm='l2')
###    
##    zui=np.max(train_data)
#####        
    train_data=(train_data-mu)/sigma
#    train_data=np.transpose(train_data)
#    train_data=np.log(train_data)
#    train_data=train_data/zui
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
    s=[]
    s1=[]
    with tf.Graph().as_default(): 
    
    
    
    
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
        
        
        
        
        def compute_accuracy(v_xs, v_ys,norm):
            global prediction
            global s
            global s1
            global on_train
            y_pre = sess.run(prediction, feed_dict={xs: v_xs,on_train: norm , keep_prob: 1,pkeep_conv:1})
            correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, on_train: norm ,keep_prob: 1,pkeep_conv:1})
            s=y_pre
            t=result
            s1.append(t)
            return result
        
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
        xs = tf.placeholder(tf.float32, [None, 6144])   # 28x28
        ys = tf.placeholder(tf.float32, [None, 4])
        
        
        
        
        
        
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
        
        
        keep_prob = tf.placeholder(tf.float32)
        x_image = tf.reshape(xs, [-1, 12, 512, 1])
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
        W_fc1 = weight_variable([3*128*64, 1024])
        b_fc1 = bias_variable([1024])
        # [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
        h_pool2_flat = tf.reshape(h_pool2, [-1, 3*128*64])
        h1=tf.matmul(h_pool2_flat, W_fc1) + b_fc1
        
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
        W_fc2 = weight_variable([1024, 4])
        b_fc2 = bias_variable([4])
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
        
        
        prediction = tf.nn.softmax(h2)
        
        
        # the error between prediction and real data
        #
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=ys,logits=prediction)
        cross_entropy = tf.reduce_mean(cross_entropy)
        train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)
        
        
        
        
        
        
        #
        #cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
        #                                              reduction_indices=[1]))       # loss
        #train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        #config = tf.ConfigProto()
        #config.gpu_options.per_process_gpu_memory_fraction = 0.4
        #session = tf.Session(config=config)
        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)  
        #sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) 
        def get_batch_data():
            [label, images] = [train_label,train_data]
            images = tf.cast(images, tf.float32)
            label = tf.cast(label, tf.float32)
            input_queue = tf.train.slice_input_producer([images, label], shuffle=False)
            image_batch, label_batch = tf.train.batch(input_queue, batch_size=512, num_threads=1, capacity=1024)
            return image_batch, label_batch
        image_batch, label_batch = get_batch_data()
        
        #train_data3=train_data[0:256,:]
        #label3=train_label[0:256,:]
        #train_data1=train_data[0:1275,:]
        #label1=train_label[0:1275,:]
        #train_data2=train_data[1276:2550,:]
        #label2=train_label[1276:2550,:]
        #train_data3=train_data[2551:3825,:]
        #label3=train_label[2551:3825,:]
        #train_data4=train_data[3826:5009,:]
        #label4=train_label[3826:5009,:]
        
        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess, coord)
        #    image_batch_v, label_batch_v = sess.run([image_batch, label_batch])
        # important step
        # tf.initialize_all_variables() no long valid from
        # 2017-03-02 if using tensorflow >= 0.12
        #   config=tf.ConfigProto()
        #   config = tf.ConfigProto() 
        #   config.gpu_options.per_process_gpu_memory_fraction = 0.4
        #   session = tf.Session(config=config)
            tf.global_variables_initializer().run()
        #    coord.request_stop()
        #    coord.join(threads)
        #   sess.graph.finalize()
        #   n=0
        
            for i in range(2500):
                image_batch_v, label_batch_v = sess.run([image_batch, label_batch])
        #      if ((i+1) % 477==0&i!=0):
        #        train_data1=train_data[0:256,:]
        #        label1=train_label[0:256,:]
        #       train_data2=train_data[1193:2385,:]
        #       label2=train_label[1193:2385,:]
        #       train_data3=train_data[2386:3578,:]
        #       label3=train_label[2386:3578,:]
        #       train_data4=train_data[3579:4771,:]
        #       label4=train_label[3579:4771,:]
        #         sess.run(train_step, feed_dict={xs: train_data[4771:4772,:], ys: train_label[4771:4772,:], keep_prob: 0.5})
        #         n=n+1
        #      else :
        #       if i % 4 ==0:
        #           sess.run(train_step, feed_dict={xs:train_data1, ys: label1, keep_prob: 0.5})
        #       elif i % 4 ==1:
        #           sess.run(train_step, feed_dict={xs:train_data2, ys: label2, keep_prob: 0.5})
        #       elif i % 4 ==2:
        #           sess.run(train_step, feed_dict={xs:train_data3, ys: label3, keep_prob: 0.5})
        #       elif i % 4 ==3:
        #           sess.run(train_step, feed_dict={xs:train_data4, ys: label4, keep_prob: 0.5})
        #       if i % 2==0:
        #        sess.run(train_step, feed_dict={xs: image_batch_v, ys:label_batch_v , keep_prob: 0.5})
        #       if i % 2==1:
        #          sess.run(train_step, feed_dict={xs: train_data2, ys: label2, keep_prob: 0.5})
               
            
        #        sess.run(train_step, feed_dict={xs: train_data, ys: train_label,on_train: True, keep_prob: 0.5,pkeep_conv:1})
                sess.run(train_step, feed_dict={xs: image_batch_v, ys:label_batch_v,on_train: True, keep_prob: 0.5,pkeep_conv:1})
        #        sess.run(train_step, feed_dict={xs: train_data3, ys:label3,on_train: True, keep_prob: 0.5,pkeep_conv:1})
                if i % 200 == 0:
                  print(compute_accuracy( image_batch_v,label_batch_v,norm=False))
                  print(compute_accuracy(test_data,test_label,norm=False))
        #        if i % 200 == 0:
        
        print('time')
        new1=os.path.join(path,str(jj))
        new2=new1+p
        new3=new1+p1
        np.save(new2,s)
        np.save(new3,s1)
#    coord.join(threads)
#           correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(ys, 1))
#           accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#           print(sess.run(accuracy, feed_dict={xs: test_data,ys:test_label,on_train: True, keep_prob: 1}))
          
#def correct(s,test_label):
#    T=np.zeros((4,4))
#    test_labelcopy=test_label
#    scopy=s
#    n=len(s)
#    for i in range(n):
#        for j in range(1,5):
#            if test_labelcopy[i,j-1]==1:
#                test_labelcopy[i,j-1]=j
#                
#    
#    #idx = np.where(test_label==1)
#    for i in range(n):
#        b=np.argmax(s[i,:])
#        scopy[i,:]=[0,0,0,0]
#        scopy[i,b]=b+1
#    #g=s[idx[0],:]
#    
#    for i in range(1,5):
#        idx = np.where(test_labelcopy==i)
#        for j in range(1,5):
#            g=scopy[idx[0],:]
#            idx1=np.where(g==j)
#            m=len(idx1[0])
#            m1=len(idx[0])
#            T[i-1,j-1]=m/m1
#    return T
              
#    np.save(r"E:\deepdata\36.npy",s)
#    np.save(r"E:\deepdata\361.npy",s1)
#        print(i)