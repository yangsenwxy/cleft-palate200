# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 20:16:07 2018

@author: Administrator
"""
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np
import tensorflow as tf


def my_confusion_matrix(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    labels = list(set(y_true))
    conf_mat = confusion_matrix(y_true, y_pred, labels = labels)
#    print ("confusion_matrix(left labels: y_true, up labels: y_pred):"
#    print "labels\t",
    for i in range(len(labels)):
        print (labels[i],"\t",)
    print 
    for i in range(len(conf_mat)):
        print (i,"\t",)
        for j in range(len(conf_mat[i])):
            print (conf_mat[i][j],'\t',)
        print 
    print
y_true=np.load(r'E:\ysdeeplearn\交叉验证四类rnnhurst_children\1s3.npy')
y_pred=np.load(r'E:\ysdeeplearn\交叉验证四类rnnhurst_children\1s.npy')
print ("classification_report(left: labels):")
ff=tf.argmax(y_pred,1)
gg=tf.argmax(y_true,1)
with tf.Session() as sess:
    ff1=sess.run(ff)+1
    ff4=sess.run(gg)+1
    conf_mat = confusion_matrix(ff4, ff1)
    print(conf_mat)
    
    print (classification_report(ff4, ff1))
from sklearn.metrics import confusion_matrix

y_true = [2, 1, 0, 1, 2, 0]
y_pred = [2, 0, 0, 1, 2, 1]

C=confusion_matrix(y_true, y_pred)
print(C, end='\n\n')


my_confusion_matrix(ff4,ff1)