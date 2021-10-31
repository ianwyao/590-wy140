# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 00:59:38 2021

@author: Weihan Yao
"""
import numpy as np
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from tensorflow.keras.models import load_model
from sklearn import metrics
from sklearn.metrics import auc,roc_curve

##Load train, test and val data
x_train = np.loadtxt('x_train.txt', dtype=int)
y_train = np.loadtxt('y_train.txt', dtype=int)
x_test = np.loadtxt('x_test.txt', dtype=int)
y_test = np.loadtxt('y_test.txt', dtype=int)
x_val = np.loadtxt('x_val.txt', dtype=int)
y_val = np.loadtxt('y_val.txt', dtype=int)

##First model evaluation
model = Sequential()
model.load_weights('cnn_model.h5')
print('Evaluate test data:')
model.evaluate(x_test, y_test)
print('Evaluate validation data:')
model.evaluate(x_val,y_val)
print('Evaluate train data:')
model.evaluate(x_train, y_train)

##Second model evaluation
model2 = Sequential()
model2.load_weights('rnn_model.h5')
print('Evaluate test data:')
model2.evaluate(x_test, y_test)
print('Evaluate validation data:')
model2.evaluate(x_val,y_val)
print('Evaluate train data:')
model2.evaluate(x_train, y_train)
