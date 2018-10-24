# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 10:01:15 2017

@author: Administrator
"""

import wave
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
import scipy.io as sio 
import numpy as np

#得到文件夹下的所有文件名称 
f=wave.open("E:\yuyinshuju\za.wav",'rb')
params = f.getparams()
nchannels, sampwidth, framerate, nframes = params[:4]
strData = f.readframes(nframes)#读取音频，字符串格式
waveData = np.fromstring(strData,dtype=np.int16)#将字符串转化为int
waveData = waveData*1.0/(max(abs(waveData)))#wave幅值归一化
# plot the wave
time = np.arange(0,nframes)*(1.0 / framerate)
plt.plot(time,waveData)
plt.xlabel("Time(s)")
plt.ylabel("Amplitude")
plt.title("Single channel wavedata")
plt.show()



load_test_label = r'E:\deeptrain\test_label826.mat'
load_data = sio.loadmat(load_test_label)
load_matrix = load_data['test_label'] #假设文件中存有字符变量是matrix，例如matlab中save(load_fn, 'matrix');当然可以保存多个save(load_fn, 'matrix_x', 'matrix_y', ...);
test_label=np.array(load_matrix,dtype='float32')




load_train_label= r'E:\deeptrain\train_label827.mat'
load_data = sio.loadmat(load_train_label)
load_matrix = load_data['train_label'] #假设文件中存有字符变量是matrix，例如matlab中save(load_fn, 'matrix');当然可以保存多个save(load_fn, 'matrix_x', 'matrix_y', ...);
train_label=np.array(load_matrix,dtype='float32')

load_train_data= r'E:\deeptrain\train_data826.mat'
load_data = sio.loadmat(load_train_data)
load_matrix = load_data['train_data'] #假设文件中存有字符变量是matrix，例如matlab中save(load_fn, 'matrix');当然可以保存多个save(load_fn, 'matrix_x', 'matrix_y', ...);
train_data=np.array(load_matrix,dtype='float32')

load_test_data= r'E:\deeptrain\test_data826.mat'
load_data = sio.loadmat(load_test_data)
load_matrix = load_data['test_data'] #假设文件中存有字符变量是matrix，例如matlab中save(load_fn, 'matrix');当然可以保存多个save(load_fn, 'matrix_x', 'matrix_y', ...);
test_data=np.array(load_matrix,dtype='float32')
train1=train_data[1:1193]


#def generate_data():
#    num = 25
#    label = np.asarray(range(0, num))
#    images = np.random.random([num, 5, 5, 3])
#    print('label size :{}, image size {}'.format(label.shape, images.shape))
#    return label, images

def get_batch_data():
    [label, images] = [train_label,train_data]
    images = tf.cast(images, tf.float32)
    label = tf.cast(label, tf.int32)
    input_queue = tf.train.slice_input_producer([images, label], shuffle=False)
    image_batch, label_batch = tf.train.batch(input_queue, batch_size=10, num_threads=1, capacity=64)
    return image_batch, label_batch

image_batch, label_batch = get_batch_data()
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)
    i = 0
    try:
        while not coord.should_stop():
            image_batch_v, label_batch_v = sess.run([image_batch, label_batch])
            i += 1
            for j in range(10):
                print(image_batch_v.shape, label_batch_v[j])
    except tf.errors.OutOfRangeError:
        print("done")
    finally:
        coord.request_stop()
    coord.join(threads)

