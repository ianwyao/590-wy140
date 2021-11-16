# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 14:34:22 2021

@author: Weihan Yao
"""

import numpy as np
import pandas as pd
import seaborn as sns
import pickle
#import h5py
import collections

import matplotlib.pyplot as plt
from keras import models
from keras import layers

from keras.layers import Conv2D
from keras.layers import Input
from keras.layers import MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.layers import BatchNormalization

#GET DATASET
from keras.datasets import mnist
from keras.datasets import fashion_mnist
from keras.datasets import cifar10
from keras.datasets import cifar100

(X, Y), (test_images, test_labels) = cifar10.load_data()
(XF, YF), (test_XF, test_YF) = cifar100.load_data()

##Remove horse pic
horse_index = np.where(Y != [7])[0].tolist()
X = X[horse_index]
horse_index_f = np.where(YF != [7])[0].tolist()
XF = XF[horse_index]

#NORMALIZE AND RESHAPE
X=X/np.max(X) 
X=X.reshape(45000,32,32,3); 

XF=XF/np.max(XF) 
XF=XF.reshape(45000,32,32,3); 

#MODEL
n_bottleneck=10
batch_size = 1000

#SHALLOW
#model = models.Sequential()
#model.add(layers.Dense(n_bottleneck, activation='linear', input_shape=(28 * 28,)))
#model.add(layers.Dense(28*28,  activation='linear'))

#Convolutional autoencoder
model = models.Sequential()

model.add(Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(BatchNormalization())     # 32x32x32
model.add(Conv2D(32, kernel_size=3, strides=2, padding='same', activation='relu'))      # 16x16x32
model.add(Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu'))      # 16x16x32
model.add(BatchNormalization())     # 16x16x32
model.add(UpSampling2D())
model.add(Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu'))      # 32x32x32
model.add(BatchNormalization())
model.add(Conv2D(3,  kernel_size=1, strides=1, padding='same', activation='sigmoid'))   # 32x32x3

model.compile(optimizer='adam', metrics=['accuracy'], loss='mean_squared_error')
model.summary()



#COMPILE AND FIT
model.compile(optimizer='rmsprop',
                loss='mean_squared_error',metrics = ['acc'])
model.summary()
history = model.fit(X, X, epochs=10, batch_size=batch_size,validation_split=0.2)

#SAVE THE MODEL AND PARAMETERS
def save_model():
    # save model and architecture to single file
    model.save("model.h5")
    print("Saved model to disk")
    f = open('parameters.pckl', 'wb')
    pickle.dump([n_bottleneck,batch_size], f)
    f.close()

#LOAD THE MODEL AND PARAMETERS
def load_model():
    with open('parameters.pckl','rb') as f:
        n_bottleneck,batch_size = pickle.load(f)
    # load weights into new model
    model.load_weights("model.h5")
    print("Loaded model from disk")

#EXTRACT MIDDLE LAYER (REDUCED REPRESENTATION)
#from keras import Model 
#extract = Model(model.inputs, model.layers[-2].output)

X1 = model.predict(X)
print(X1.shape)

X2 = model.predict(XF)
print(X2.shape)

#2D PLOT
plt.scatter(X1[:,0], X1[:,1], c=Y, cmap='tab10')
plt.show()

#3D PLOT
ax = plt.figure(figsize=(16,10)).gca(projection='3d')
ax.scatter(
    xs=X1[:,0], 
    ys=X1[:,1], 
    zs=X1[:,2], 
    c=Y, 
    cmap='tab10'
)
plt.show()

#PLOT ORIGINAL AND RECONSTRUCTED 
#X1=model.predict(X) 

#RESHAPE
X=X.reshape(45000,32,32,3); #print(X[0])
X1=X1.reshape(45000,32,32,3); #print(X[0])

#COMPARE ORIGINAL 
f, ax = plt.subplots(4,1)
I1=11; I2=46
ax[0].imshow(X[I1])
ax[1].imshow(X1[I1])
ax[2].imshow(X[I2])
ax[3].imshow(X1[I2])
plt.show()

##Plot accuracy and loss
acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([0,1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('mean square error')
plt.ylim([0,0.1])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

##Reshape back to the original
X = X.reshape(45000,32*32*3)
X1 = X1.reshape(45000,32*32*3)
XF = XF.reshape(45000,32*32*3)
X2 = X2.reshape(45000, 32*32*3)

##Anamoly detection
##Calculate mean square error of XF
A = np.subtract(X1,X)
B = A*A
print(B.shape)
mse = np.sum(B)/len(X)
print('Mean Sqaure Error of MNIST:',mse)

##Loop through Mnist-fashion data:
AF = np.subtract(X2,XF)
BF = AF*AF

column_sum = BF.sum(axis=1)
label = ['anomaly' if item > mse else 'normal' for item in column_sum]
##counter=collections.Counter(label)
print('anomaly ratio: ',label.count('anomaly')/len(label))
