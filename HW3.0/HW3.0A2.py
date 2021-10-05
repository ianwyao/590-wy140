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
from keras.datasets import imdb
from tensorflow.keras import optimizers
import tensorflow
#----------------------------------------
#GET DATA
#----------------------------------------

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
num_words=10000)

##Training data and labels
train_data[0]
train_labels[0]

##Not exceed 10000
max([max(sequence) for sequence in train_data])

##Encoding
def vectorize_sequences(sequences, dimension=10000):
	results = np.zeros((len(sequences), dimension))
	for i, sequence in enumerate(sequences):
		results[i, sequence] = 1.
	return results

##Partition data
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

##Add layers:
##relu + sigmoid
##Add L1 regularization
kernel_reg = 0.01
lr = 0.001
epochs = 20
loss = 'binary_crossentropy'


model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid',kernel_regularizer=tensorflow.keras.regularizers.l1(kernel_reg)))

##Learning rate = 0.001
##Loss function: binary cross entropy
model.compile(optimizer = optimizers.RMSprop(lr = lr),
			  loss=loss,
			  metrics=['accuracy'])

##epochs: 20
history = model.fit(partial_x_train,
					partial_y_train,
					epochs=epochs,
					batch_size=len(partial_x_train),
					validation_data=(x_val, y_val))

history_dict = history.history

iplot = True
if (iplot == True):
	##Plot training and validation loss
	history_dict = history.history
	loss_values = history_dict['loss']
	val_loss_values = history_dict['val_loss']
	epochs = range(1, len(loss_values) + 1)
	plt.plot(epochs, loss_values, 'bo', label='Training loss')
	plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
	plt.title('Training and validation loss')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.legend()
	plt.show()

    ##Plot training and validation accuracy
	acc_values = history_dict['accuracy']
	val_acc_values = history_dict['val_accuracy']
	plt.plot(epochs, acc_values, 'bo', label='Training acc')
	plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
	plt.title('Training and validation accuracy')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.legend()
	plt.show()

##Retrain the data and evaluate
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid',kernel_regularizer=tensorflow.keras.regularizers.l1(kernel_reg)))
model.compile(optimizer=optimizers.RMSprop(lr = lr),
			  loss='binary_crossentropy',
			  metrics=['accuracy'])
model.fit(x_train, y_train, epochs=20, batch_size=len(x_train))
results = model.evaluate(x_test, y_test)
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