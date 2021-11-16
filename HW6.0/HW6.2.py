# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 20:03:37 2021

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

#GET DATASET
from keras.datasets import mnist
from keras.datasets import fashion_mnist
(X, Y), (test_images, test_labels) = mnist.load_data()
(XF, YF), (test_XF, test_YF) = fashion_mnist.load_data()

#NORMALIZE AND RESHAPE
X=X/np.max(X) 
X=X.reshape(60000,28,28,1); 

XF=XF/np.max(XF) 
XF=XF.reshape(60000,28,28,1); 

#MODEL
n_bottleneck=10
batch_size = 1000

#SHALLOW
#model = models.Sequential()
#model.add(layers.Dense(n_bottleneck, activation='linear', input_shape=(28 * 28,)))
#model.add(layers.Dense(28*28,  activation='linear'))

#Convolutional autoencoder
input_img = Input(shape=(28,28,1))
enc_conv1 = Conv2D(12, (3, 3), activation='relu', padding='same')(input_img)
enc_pool1 = MaxPooling2D((2, 2), padding='same')(enc_conv1)
enc_conv2 = Conv2D(8, (4, 4), activation='relu', padding='same')(enc_pool1)
enc_ouput = MaxPooling2D((4, 4), padding='same')(enc_conv2)

dec_conv2 = Conv2D(8, (4, 4), activation='relu', padding='same')(enc_ouput)
dec_upsample2 = UpSampling2D((4, 4))(dec_conv2)
dec_conv3 = Conv2D(12, (3, 3), activation='relu')(dec_upsample2)
dec_upsample3 = UpSampling2D((2, 2))(dec_conv3)
dec_output = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(dec_upsample3)

model = Model(input_img, dec_output)
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
X=X.reshape(60000,28,28); #print(X[0])
X1=X1.reshape(60000,28,28); #print(X[0])

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
X = X.reshape(60000,28*28)
X1 = X1.reshape(60000,28*28)
XF = XF.reshape(60000,28*28)
X2 = X2.reshape(60000, 28*28)

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
