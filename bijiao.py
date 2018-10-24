# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 21:40:17 2018

@author: Administrator
"""

from tensorflow.contrib import rnn
import math
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
pred=np.load(r'E:\ysdeeplearn\敏感音\mel\1s.npy')
y=np.load(r'E:\ysdeeplearn\敏感音\mel\1s3.npy')
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
ll= tf.cast(correct_pred, tf.float32)

with tf.Session() as sess:
    ll=sess.run(ll)