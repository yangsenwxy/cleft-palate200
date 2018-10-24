# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 16:21:33 2017

@author: Administrator
"""

from __future__ import print_function
import tensorflow as tf
import scipy.io as sio 
import numpy as np
import os

path = r'E:\ysdeeplearn\deepcode\deeptrain\sp256\基于人的'
dirs = os.listdir( path )
l1='test_label.mat'
l2='train_label.mat'
l3='test_data.mat'
l4='train_data.mat'
path1 = "E:\ysdeeplearn\结果成人\\"
L=np.zeros((2,2))
L1=np.zeros((2,2))
s4=np.zeros((1,1))

p='s.npy'
p1='s1.npy'
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
    test_label1=np.array(load_matrix,dtype='float32')




    load_train_label= r'E:\deeptrain\train_label827.mat'
    load_data = sio.loadmat(newDir2)
    load_matrix = load_data['train_label'] #假设文件中存有字符变量是matrix，例如matlab中save(load_fn, 'matrix');当然可以保存多个save(load_fn, 'matrix_x', 'matrix_y', ...);
    train_label1=np.array(load_matrix,dtype='float32')

    load_train_data= r'E:\deeptrain\train_data826.mat'
    load_data = sio.loadmat(newDir4)
    load_matrix = load_data['train_data'] #假设文件中存有字符变量是matrix，例如matlab中save(load_fn, 'matrix');当然可以保存多个save(load_fn, 'matrix_x', 'matrix_y', ...);
    train_data=np.array(load_matrix,dtype='float32')

    load_test_data= r'E:\deeptrain\test_data826.mat'
    load_data = sio.loadmat(newDir3)
    load_matrix = load_data['test_data'] #假设文件中存有字符变量是matrix，例如matlab中save(load_fn, 'matrix');当然可以保存多个save(load_fn, 'matrix_x', 'matrix_y', ...);
    test_data=np.array(load_matrix,dtype='float32')
    
    f1=len(train_label1)
    train_label=np.zeros((f1,2))
    f2=len(test_label1)
    test_label=np.zeros((f2,2))
    for i in range(f1):
        if train_label1[i,0]==1:
            train_label[i,:]=[1,0]
        else:
            train_label[i,:]=[0,1]
    
    for j in range(f2):
        if test_label1[j,0]==1:
            test_label[j,:]=[1,0]
        else:
            test_label[j,:]=[0,1]
#    mu = np.mean(train_data,axis = 0)
#    sigma = np.std(train_data,axis = 0)
    
#    zui=np.max(train_data)
#    train_data=train_data/zui
    #    
#    train_data=(train_data-mu)/sigma
    
#    mu1 = np.mean(test_data,axis = 0)
#    sigma1 = np.std(test_data,axis = 0)
    
#    zui1=np.max(test_data)
#    test_data=test_data/zui1
    #    
#    test_data=(test_data-mu1)/sigma1

    s1=[]
    s2=[]
    #load_test_label = r'E:\ysdeeplearn\deepcode\deeptrain\mfcc\cross\36\test_label.mat'
    #load_data = sio.loadmat(load_test_label)
    #load_matrix = load_data['test_label'] #假设文件中存有字符变量是matrix，例如matlab中save(load_fn, 'matrix');当然可以保存多个save(load_fn, 'matrix_x', 'matrix_y', ...);
    #test_label=np.array(load_matrix,dtype='float32')
    #
    #
    #
    #
    #load_train_label= r'E:\ysdeeplearn\deepcode\deeptrain\mfcc\cross\36\train_label.mat'
    #load_data = sio.loadmat(load_train_label)
    #load_matrix = load_data['train_label'] #假设文件中存有字符变量是matrix，例如matlab中save(load_fn, 'matrix');当然可以保存多个save(load_fn, 'matrix_x', 'matrix_y', ...);
    #train_label=np.array(load_matrix,dtype='float32')
    #
    #load_train_data= r'E:\ysdeeplearn\deepcode\deeptrain\mfcc\cross\36\train_data.mat'
    #load_data = sio.loadmat(load_train_data)
    #load_matrix = load_data['train_data'] #假设文件中存有字符变量是matrix，例如matlab中save(load_fn, 'matrix');当然可以保存多个save(load_fn, 'matrix_x', 'matrix_y', ...);
    #train_data=np.array(load_matrix,dtype='float32')
    #
    #load_test_data= r'E:\ysdeeplearn\deepcode\deeptrain\mfcc\cross\36\test_data.mat'
    #load_data = sio.loadmat(load_test_data)
    #load_matrix = load_data['test_data'] #假设文件中存有字符变量是matrix，例如matlab中save(load_fn, 'matrix');当然可以保存多个save(load_fn, 'matrix_x', 'matrix_y', ...);
    #test_data=np.array(load_matrix,dtype='float32')
    # number 1 to 10 data
    
    with tf.Graph().as_default(): 
        def compute_accuracy(v_xs, v_ys,norm):
            global prediction
            global s2
            global s1
            global on_train
            y_pre = sess.run(prediction, feed_dict={xs: v_xs,on_train: norm , keep_prob: 1})
            correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, on_train: norm ,keep_prob: 1})
            s2=y_pre
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
        on_train = tf.placeholder(tf.bool)
        xs = tf.placeholder(tf.float32, [None, 6144])   # 28x28
        ys = tf.placeholder(tf.float32, [None, 2])
        
        
        
        
        
        
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
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        keep_prob = tf.placeholder(tf.float32)
        x_image = tf.reshape(xs, [-1, 12, 512, 1])
        #x_image=tf.transpose(x_image)
        #x_image=tf.image.resize_images(x_image1,[16,240],method=0)
        # print(x_image.shape)  # [n_samples, 28,28,1]
        
        ## conv1 layer ##
        W_conv1 = weight_variable([5,5, 1,32]) # patch 5x5, in size 1, out size 32
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 28x28x32
        h_pool1 = max_pool_2x2(h_conv1)                                         # output size 14x14x32
        
        ## conv2 layer ##
        W_conv2 = weight_variable([5,5, 32, 64]) # patch 5x5, in size 32, out size 64
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 14x14x64
        h_pool2 = max_pool_2x2(h_conv2)                                         # output size 7x7x64
        
        ## fc1 layer ##
        W_fc1 = weight_variable([3*128*64, 1024])
        b_fc1 = bias_variable([1024])
        # [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
        h_pool2_flat = tf.reshape(h_pool2, [-1, 3*128*64])
#        h1=tf.matmul(h_pool2_flat, W_fc1) + b_fc1
#        
#        fc_mean, fc_var = tf.nn.moments(
#                        h1,
#                        axes=[0,],   # the dimension you wanna normalize, here [0] for batch
#                                    # for image, you wanna do [0, 1, 2] for [batch, height, width] but not channel
#                    )
#        scale = tf.Variable(tf.ones([1024]))
#        shift = tf.Variable(tf.zeros([1024]))
#        epsilon = 0.001
#        
#                    # apply moving average for mean and var when train on batch
#        ema = tf.train.ExponentialMovingAverage(decay=0.5)
#        def mean_var_with_update():
#            ema_apply_op = ema.apply([fc_mean, fc_var])
#            with tf.control_dependencies([ema_apply_op]):
#                return tf.identity(fc_mean), tf.identity(fc_var)
#        mean, var = mean_var_with_update()
#        Wx_plus_b = tf.nn.batch_normalization(h1, mean, var, shift, scale, epsilon)
        
        
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
        ema = tf.train.ExponentialMovingAverage(decay=0.90)
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
        
        #3 layer
        
        
        W_fc3 = weight_variable([1024, 1024])
        b_fc3 = bias_variable([1024])
        h3=tf.matmul(h_fc1_drop, W_fc3) + b_fc3
        
#        fc_mean2, fc_var2 = tf.nn.moments(
#                        h3,
#                        axes=[0],   # the dimension you wanna normalize, here [0] for batch
#                                    # for image, you wanna do [0, 1, 2] for [batch, height, width] but not channel
#                    )
#        scale2 = tf.Variable(tf.ones([1024]))
#        shift2 = tf.Variable(tf.zeros([1024]))
#        epsilon2 = 0.001
#        
#                    # apply moving average for mean and var when train on batch
#        ema = tf.train.ExponentialMovingAverage(decay=0.9)
#        def mean_var_with_update():
#            ema_apply_op = ema.apply([fc_mean2, fc_var2])
#            with tf.control_dependencies([ema_apply_op]):
#                return tf.identity(fc_mean2), tf.identity(fc_var2)
#        mean2, var2 = mean_var_with_update()
#        Wx_plus_b2 = tf.nn.batch_normalization(h3, mean2, var2, shift2, scale2, epsilon2)
        
        fc_mean2, fc_var2 = tf.nn.moments(
                        h3,
                        axes=[0],   # the dimension you wanna normalize, here [0] for batch
                                    # for image, you wanna do [0, 1, 2] for [batch, height, width] but not channel
                    )
        scale2 = tf.Variable(tf.ones([1024]))
        shift2 = tf.Variable(tf.zeros([1024]))
        epsilon2 = 0.001
        
                    # apply moving average for mean and var when train on batch
        ema = tf.train.ExponentialMovingAverage(decay=0.90)
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
#        W_fc2 = weight_variable([1024, 2])
#        b_fc2 = bias_variable([2])
#        h2=tf.matmul(h_fc2_drop, W_fc2) + b_fc2
#        fc_mean1, fc_var1 = tf.nn.moments(
#                        h2,
#                        axes=[0],   # the dimension you wanna normalize, here [0] for batch
#                                    # for image, you wanna do [0, 1, 2] for [batch, height, width] but not channel
#                    )
#        scale1 = tf.Variable(tf.ones([2]))
#        shift1 = tf.Variable(tf.zeros([2]))
#        epsilon1 = 0.001
#        
#                    # apply moving average for mean and var when train on batch
#        ema = tf.train.ExponentialMovingAverage(decay=0.9)
#        def mean_var_with_update():
#            ema_apply_op = ema.apply([fc_mean1, fc_var1])
#            with tf.control_dependencies([ema_apply_op]):
#                return tf.identity(fc_mean1), tf.identity(fc_var1)
#        mean1, var1 = mean_var_with_update()
#        Wx_plus_b1 = tf.nn.batch_normalization(h2, mean1, var1, shift1, scale1, epsilon1)
        
        
        prediction = tf.nn.softmax(h2)
        
        
        # the error between prediction and real data
        #
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=ys,logits=prediction)
        cross_entropy = tf.reduce_mean(cross_entropy)
        train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)
        
        
        
        def get_batch_data():
            [label, images] = [train_label,train_data]
            images = tf.cast(images, tf.float32)
            label = tf.cast(label, tf.float32)
            input_queue = tf.train.slice_input_producer([images, label], shuffle=False)
            image_batch, label_batch = tf.train.batch(input_queue, batch_size=512, num_threads=1, capacity=1024)
            return image_batch, label_batch
        image_batch, label_batch = get_batch_data()
        
        
        #
        #cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
        #                                              reduction_indices=[1]))       # loss
        #train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        #config = tf.ConfigProto()
        #config.gpu_options.per_process_gpu_memory_fraction = 0.4
        #session = tf.Session(config=config)
        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)  
        #sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) 
        #def get_batch_data():
        #    [label, images] = [train_label,train_data]
        #    images = tf.cast(images, tf.float32)
        #    label = tf.cast(label, tf.float32)
        #    input_queue = tf.train.slice_input_producer([images, label], shuffle=False)
        #    image_batch, label_batch = tf.train.batch(input_queue, batch_size=1000, num_threads=1, capacity=1300)
        #    return image_batch, label_batch
        #image_batch, label_batch = get_batch_data()
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
        
            for i in range(3000):
                image_batch_v, label_batch_v = sess.run([image_batch, label_batch])
        #        image_batch_v, label_batch_v = sess.run([image_batch, label_batch])
        #      if ((i+1) % 477==0&i!=0):
        #       train_data1=train_data[0:1192,:]
        #       label1=train_label[0:1192,:]
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
               
            
                sess.run(train_step, feed_dict={xs: image_batch_v, ys: label_batch_v,on_train: True, keep_prob: 0.5})
                if i % 200 == 0:
                  print(compute_accuracy(train_data,train_label,norm=False))
                  print(compute_accuracy(test_data,test_label,norm=False))
                  
            print('time')
            new1=os.path.join(path1,str(s))
            new2=new1+p
            new3=new1+p1
            np.save(new2,s1)
            np.save(new3,s2)
            
              
#              
#    def correct(s,test_label):
#        T=np.zeros((2,2))
#        test_labelcopy=test_label
#        scopy=s
#        n=len(s)
#        for i in range(n):
#            for j in range(1,3):
#                if test_labelcopy[i,j-1]==1:
#                    test_labelcopy[i,j-1]=j
#                    
#        
#        #idx = np.where(test_label==1)
#        for i in range(n):
#            b=np.argmax(s[i,:])
#            scopy[i,:]=[0,0]
#            scopy[i,b]=b+1
#        #g=s[idx[0],:]
#        
#        for i in range(1,3):
#            idx = np.where(test_labelcopy==i)
#            for j in range(1,3):
#                g=scopy[idx[0],:]
#                idx1=np.where(g==j)
#                m=len(idx1[0])
#                m1=len(idx[0])
#                T[i-1,j-1]=m/m1
#        return T
##    L=correct(s1,test_label)
#    s4=s2+s4
#    pathsave=r'E:\deepdata\jiyuren'
#    v1='two.npy'
#    v2='one.npy'
#    new1=os.path.join(pathsave,s) 
##    new1=pathsave+s
#    new2=new1+v1
#    new3=new1+v2
##    pathsave1=r"E:\deepdata\"
##    new1=os.path.join(pathsave,s) 
##    new2=os.path.join(new1,v1) 
##    new3=os.path.join(new1,v2)              
##    np.save(new2,L)
#    np.save(new3,s2)
#    L1=L1+L
##L=L
#s4=s4
##
#pathsave5=r'E:\deepdata\hui\two.npy'
#pathsave6=r'E:\deepdata\hui\one.npy'
##v1='1.npy'
##v2='one.npy'
#
#np.save(pathsave5,L1)
#np.save(pathsave6,s4)

#        print(i)