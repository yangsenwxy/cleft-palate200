# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 19:56:44 2018

@author: Administrator
"""
import numpy as np
train_data=np.array([[1,2],[4,5],[7,8]])

train_data=np.transpose(train_data)
mu = np.mean(train_data,axis=0)
sigma = np.std(train_data,axis=0)
train_data=(train_data-mu)/sigma
train_data=np.transpose(train_data)