# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 17:00:34 2021

@author: Weihan Yao
"""
##CLEAN THREE NOVELS
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

##Read three novels and select only first 250 paragraphs 
texts = []                
f = open('ascanio.txt',encoding="utf-8")
texts.append(f.read().split('\n\n'))
f.close()
txt1 = [p for i in texts for p in i if p != ''][0:250]
label1 = [1] * 250

texts = []                
f = open('road.txt',encoding="utf-8")
texts.append(f.read().split('\n\n'))
f.close()
txt2 = [p for i in texts for p in i if p != ''][0:250]
label2 = [2] * len(txt2)

texts = []                
f = open('pioneer.txt',encoding="utf-8")
texts.append(f.read().split('\n\n'))
f.close()
txt3 = [p for i in texts for p in i if p != ''][0:250]
label3 = [3] * len(txt3)

##Concatenate three texts and labels
texts = txt1 + txt2 + txt3
labels  = label1 + label2 + label3

##Hyper parameters
maxlen = 100
training_samples = 200
test_samples = 150
validation_samples = 400
max_words = 10000

##Tokenize and convert texts to sequences
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

##Pad sequences and then split text data
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=maxlen)
labels = np.asarray(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]
x_test = data[training_samples + validation_samples: training_samples + validation_samples + test_samples]
y_test = labels[training_samples + validation_samples: training_samples + validation_samples + test_samples]

##Save train, val and test data
np.savetxt('x_train.txt', x_train, fmt='%d')
np.savetxt('y_train.txt', y_train, fmt='%d')
np.savetxt('x_val.txt', x_val, fmt='%d')
np.savetxt('y_val.txt', y_val, fmt='%d')
np.savetxt('x_test.txt', x_test, fmt='%d')
np.savetxt('y_test.txt', y_test, fmt='%d')

##Save word index as pickle
with open('word_index'+'.pkl', 'wb') as f:
    pickle.dump(word_index, f, pickle.HIGHEST_PROTOCOL)

