# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 18:31:25 2021

@author: Weihan Yao
"""
import numpy as np
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, SimpleRNN
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import tensorflow
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback

##Load train, test and val data
x_train = np.loadtxt('x_train.txt', dtype=int)
y_train = np.loadtxt('y_train.txt', dtype=int)
x_test = np.loadtxt('x_test.txt', dtype=int)
y_test = np.loadtxt('y_test.txt', dtype=int)
x_val = np.loadtxt('x_val.txt', dtype=int)
y_val = np.loadtxt('y_val.txt', dtype=int)

##Parsing the Glove word-embeddings file
embeddings_index = {}
f = open('glove.6B.100d.txt',encoding = 'utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))

##Hyper parameters
max_words = 10000
embedding_dim = 100
maxlen = 100
kernel_reg = 0.01

with open('word_index' + '.pkl', 'rb') as f:
    word_index = pickle.load(f)

##Preparing the Glove word-embeddings matrix    
embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            
##CNN model definition
model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Conv1D(1,2,kernel_initializer = 'ones',bias_initializer = 'zeros', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid',kernel_regularizer=tensorflow.keras.regularizers.l2(kernel_reg)))
print(model.summary())

##Loading pretrained word embeddings into the Embedding layer
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

##Training
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=len(x_train),
                    validation_data=(x_val, y_val))

##Saving the weights of the model
model.save_weights("cnn_model.h5")

##Plotting the result
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

##RNN model definition
model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(SimpleRNN(32))
model.add(Dense(1, activation='sigmoid',kernel_regularizer=tensorflow.keras.regularizers.l2(kernel_reg)))
print(model.summary())

##Loading pretrained word embeddings into the Embedding layer
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

##Training
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=len(x_train),
                    validation_data=(x_val, y_val))

##Saving the weights of the model
model.save_weights("rnn_model.h5")

##Plotting the result
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


