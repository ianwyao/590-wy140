# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 09:14:41 2021

@author: Weihan Yao
"""

import pandas  as  pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from keras import models
from keras import layers
from keras.utils.np_utils import to_categorical
from keras.datasets import reuters
from keras import models
from keras import layers
from tensorflow.keras import optimizers
import tensorflow
#----------------------------------------
#GET DATA
#----------------------------------------

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(
num_words=10000)

##Check the length of training data and test data
len(train_data)
len(test_data)

##Vectorize training data and test data
def vectorize_sequences(sequences, dimension=10000):
	results = np.zeros((len(sequences), dimension))
	for i, sequence in enumerate(sequences):
		results[i, sequence] = 1.
	return results
	
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

##One hot encoding for training labels and test labels
def to_one_hot(labels, dimension=46):
	results = np.zeros((len(labels), dimension))
	for i, label in enumerate(labels):
		results[i, label] = 1.
	return results
	
one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels = to_one_hot(test_labels)

##Hyper parameters
##relu + softmax
##Add L2 regularization
##Use categorical cross entropy
kernel_reg = 0.01
lr = 0.005
epochs = 15
loss = 'categorical_crossentropy'

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax',kernel_regularizer=tensorflow.keras.regularizers.l2(kernel_reg)))

model.compile(optimizer = optimizers.RMSprop(lr = lr),
			  loss= loss,
			  metrics=['accuracy'])

##Partition the data
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]


history = model.fit(partial_x_train,
					partial_y_train,
					epochs=epochs,
					batch_size=len(partial_x_train),
					validation_data=(x_val, y_val))

iplot = True
if (iplot == True):
	##Plot training and validation loss
	loss = history.history['loss']
	val_loss = history.history['val_loss']
	epochs = range(1, len(loss) + 1)
	plt.plot(epochs, loss, 'bo', label='Training loss')
	plt.plot(epochs, val_loss, 'b', label='Validation loss')
	plt.title('Training and validation loss')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.legend()
	plt.show()
	
	##Plot training and validation accuracy
	acc = history.history['accuracy']
	val_acc = history.history['val_accuracy']
	plt.plot(epochs, acc, 'bo', label='Training acc')
	plt.plot(epochs, val_acc, 'b', label='Validation acc')
	plt.title('Training and validation accuracy')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.legend()
	plt.show()

##Retrain the data and evaluate
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax',kernel_regularizer=tensorflow.keras.regularizers.l2(kernel_reg)))
model.compile(optimizer=optimizers.RMSprop(lr = lr),
			  loss='categorical_crossentropy',
			  metrics=['accuracy'])
model.fit(x_train,
		  one_hot_train_labels,
		  epochs=15,
		  batch_size=len(x_train),
		  validation_data=(x_val, y_val))
results = model.evaluate(x_test, one_hot_test_labels)
print('loss and accuracy : ',results)

##Prediction
yp=model.predict(x_train)
yp_val=model.predict(x_val)
yp_test=model.predict(x_test) 

##Print all three parts
print('Training data:')
print(partial_x_train[0:10])
print('Predicted training data:')
print(yp[0:10])
print('Test data:')
print(x_test[0:10])
print('Predicted test data:')
print(yp_test[0:10])
print('Validation data:')
print(x_val[0:10])
print('Predicted validation data:')
print(yp_val[0:10])




##exit()

