# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 16:48:43 2018

@author: Administrator
"""
import tensorflow as tf
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
import scipy.io as sio 
import os
import random as rd
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

train_data=np.load(r'E:\ysdeeplearn\deepcode\deeptrain\RNN训练完的特征\cepstrum\traindatatezheng.npy')
test_data=np.load(r'E:\ysdeeplearn\deepcode\deeptrain\RNN训练完的特征\cepstrum\testdatatezheng.npy')

test_label1=np.load(r'E:\ysdeeplearn\deepcode\deeptrain\RNN训练完的特征\cepstrum\testlabeltezheng.npy')
train_label1=np.load(r'E:\ysdeeplearn\deepcode\deeptrain\RNN训练完的特征\cepstrum\trainlabeltezheng.npy')


ff=tf.argmax(train_label1,1)
dd=tf.argmax(test_label1,1)
with tf.Session() as sess:
    train_label,test_label=sess.run([ff,dd])


#    from sklearn.naive_bayes import MultinomialNB 
#    model =LogisticRegression()
#        model = svm.SVC(C=0.8, kernel='rbf', gamma=20)
#        from sklearn.naive_bayes import GaussianNB
#    model = RandomForestClassifier()
    model = GaussianNB()
#    model = tree.DecisionTreeClassifier()
#    model=AdaBoostClassifier()
#    model = GradientBoostingClassifier()
##        model=ExtraTreesClassifier()
##        model=LinearDiscriminantAnalysis()
##        model=QuadraticDiscriminantAnalysis()
##        model = MultinomialNB(alpha=0.01)  
#
##        model.fit(train_x, train_y)  
#    model = KNeighborsClassifier(n_neighbors = 90)
    model.fit(train_data, train_label) 
    y_predict = model.predict(test_data) 
    acc=accuracy_score(test_label, y_predict)
    

    
#from sklearn.manifold import TSNE
#
#
#
#
#import matplotlib.pyplot as plt
#
#
#X_tsne = TSNE(learning_rate=0.5).fit_transform(test_data)
#plt.figure(figsize=(10, 5))
#plt.subplot(121)
#plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=test_label)


    
    
    
    