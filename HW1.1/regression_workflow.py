# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 21:32:36 2021

@author: Weihan Yao
"""

##Import packages
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import json
import math
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize

##Create a Data Class that opens the json file and visualizes weight and age
class Data:
    #Opening JSON file
    with open('weight.json') as json_file:
        dat = json.load(json_file)
        
    ##Define three variables   
    x = dat['x']
    y = dat['is_adult']
    z = dat['y']
    
    ##Visualize two plots, weight(age) and is_adult(weight) plots
    def visual(self,x,y,z):
        plt.ion() 
        fig,ax = plt.subplots(1,2,figsize=(30,15))
        ax[0].plot(x,z,'bo')
        ax[1].plot(z,y,'go')
        ax[0].set_xlabel('age(years)')
        ax[0].set_ylabel('weight(lb)')
        ax[1].set_xlabel('weight(lb)')
        ax[1].set_ylabel('Adult = 1 Child = 0')
        plt.show()
        
##Choosing a measure of success:
##Mean Square Error(MSE)

##Deciding on an evaluation protocol
##Break the data into 80% training 20% validation set
##Create an instance and visualize the data
data = Data()
data.visual(data.x,data.y,data.z)
x_train, x_test, y_train, y_test = train_test_split(data.x, data.z, test_size=0.2, random_state=40)

# We use standard scaler to transform data
# = (x - μ) / σ
xs_train = (x_train - np.mean(x_train))/np.std(x_train)
xs_test = (x_test - np.mean(x_test))/np.std(x_test)
ys_train = (y_train - np.mean(y_train))/np.std(y_train)
ys_test = (y_test - np.mean(y_test))/np.std(y_test)

## if age < 18, then the normalized age should less than -1.18557
NORMAL_AGE_18 = -1.18557


##Put the training data and validation(test) data in dataframes
df = pd.DataFrame()
df['x_train'] = xs_train
df['y_train'] = ys_train

df_test = pd.DataFrame()
df_test['x_test'] = xs_test
df_test['y_test'] = ys_test

##We just train those who are under 18 for the first model
df1 = df[df['x_train'] < -1.18557]
df2 = df_test[df_test['x_test'] < -1.18557]

#The first model: linear regression 
#FUNCTION TO OPTIMZE
#x[0] = k * x[1] + b
def f1(x):
    out = 0
    i = 0
    while i < len(df1):
        ##Loss function: MSE
        out += ((df1.iloc[i,0] * x[0] + x[1]) - df1.iloc[i,1])**2
        i += 1  
    mse = out/len(df1)
    return mse

##Here we could get optimizer iterations and corresponding loss value
history1 = []
def callback1(x):
    fobj = f1(x)
    history1.append(fobj)

##Define the same functions for the validation data    
def f1_val(x):
    out = 0
    i = 0
    while i < len(df2):
        out += ((df2.iloc[i,0] * x[0] + x[1]) - df2.iloc[i,1])**2
        i += 1  
    mse = out/len(df2)
    return mse

history1_val = []
def callback1_val(x):
    fobj = f1_val(x)
    history1_val.append(fobj)

##Iteration history for the training data
result = minimize(f1, [0,0], method='Nelder-Mead', tol=1e-6, callback=callback1)
print(history1)

##Iteration history for the validation data
result_val = minimize(f1_val, [0,0], method='Nelder-Mead', tol=1e-6, callback=callback1_val)
print(history1_val)

##Visualize training loss and validation loss
fig,ax = plt.subplots(1,1)
ax.plot(history1,'bo',label = 'training loss')
ax.plot(history1_val,'ro',label = 'validation loss')
ax.set_xlabel('optimizer iterations')
ax.set_ylabel('loss')
ax.legend()

#INITIAL GUESS 
#k = 0, b = 0 
#Nelder-Mead method gives a better result
print("INITIAL GUESS: xo= 0,0", " f1(xo)=",f1([0,0]))
res = minimize(f1,[0,0],method='Nelder-Mead', tol=1e-5)
popt = res.x
print("OPTIMAL PARAM:",popt)
##OPTIMAL PARAM: [5.62787628 6.11367938]

##Plot training set, validation set and the fitted model 
##The x label and y label are normalized age and normalized weight
fig,ax = plt.subplots(1,1)
ax.plot(df['x_train'], df['y_train'], 'o',label = 'training set')
xl = np.linspace(-1.8,-0.9,3)
ax.plot(xl, popt[0]*xl + popt[1],'r',label = 'model')
ax.plot(df_test['x_test'], df_test['y_test'],'gx', label = 'validation set')
ax.set_xlabel('normalized age')
ax.set_ylabel('normalized weight')
ax.legend()


##The second model: sigmoid regression
##Parameters: A,w,x0,s
##FUNCTION TO OPTIMIZE
def f2(x):
    out = 0
    i = 0
    while i < len(df):
        ##Loss function: MSE
        out += (x[0]/(1 + math.e**(-(df.iloc[i,0] - x[2])/x[1])) + x[3] - df.iloc[i,1])**2
        i += 1  
    mse = out/len(df)
    return mse

##Here we could get optimizer iterations and corresponding loss value
history2 = []
def callback2(x):
    fobj = f2(x)
    history2.append(fobj)

##Define the same functions for the validation data  
def f2_val(x):
    out = 0
    i = 0
    while i < len(df_test):
        out += (x[0]/(1 + math.e**(-(df_test.iloc[i,0] - x[2])/x[1])) + x[3] - df_test.iloc[i,1])**2
        i += 1  
    mse = out/len(df_test)
    return mse

history2_val = []
def callback2_val(x):
    fobj = f2_val(x)
    history2_val.append(fobj)

##Iteration history for the training data
##BFGS method gives a better result
result = minimize(f2, [1,1,0,0], method='BFGS', tol=1e-6, callback=callback2)
print(history2)

##Iteration history for the validation data
result_val = minimize(f2_val, [1,1,0,0], method='BFGS', tol=1e-6, callback=callback2_val)
print(history2_val)

##Visualize training loss and validation loss
fig,ax = plt.subplots(1,1)
ax.plot(history2,'bo',label = 'training loss')
ax.plot(history2_val,'ro',label = 'validation loss')
ax.set_xlabel('optimizer iterations')
ax.set_ylabel('loss')
ax.legend()

#INITIAL GUESS 
#A = 1, w = 1, x0 = 0, s = 0
print("INITIAL GUESS: xo= 1,1,0,0", " f2(xo)=",f2([1,1,0,0]))
res = minimize(f2,[1,1,0,0],method='BFGS', tol=1e-5)
popt = res.x
print("OPTIMAL PARAM:",popt)
##OPTIMAL PARAM: [ 3.93620625  0.13499955 -1.35764465 -3.51406333]

##Plot training set, validation set and the fitted model 
##The x label and y label are normalized age and normalized weight
fig,ax = plt.subplots(1,1)
ax.plot(df['x_train'], df['y_train'], 'o',label = 'training set')
xl = np.linspace(-2,1.5,100)
ax.plot(xl, popt[0]/(1 + math.e**(-(xl - popt[2])/popt[1])) + popt[3],'r',label = 'model')
ax.plot(df_test['x_test'], df_test['y_test'],'gx', label = 'validation set')
ax.set_xlabel('normalized age')
ax.set_ylabel('normalized weight')
ax.legend()

##Unnormalize data
## x = s_x * x’ + u_x
## ypred = s_y * ypred' + u_y
##I use y_train, y_test as input
##The output: is_adult

##Create a new list is_adult : 1, is_not_adult : 0
is_adult = [1 if p >= 18 else 0 for p in x_train]
##Same for the validation data
is_adult_test = [1 if p >= 18 else 0 for p in x_test]

##The third model: logistic regression model
##Parameters: w, b
##Since A = 1, s = 0 are fixed
##FUNCTION TO OPTIMIZE
def f3(x):
    out = 0
    i = 0
    while i < len(y_train):
        out += (1/(1 + math.e**(-(x[0]*y_train[i] + x[1]))) - is_adult[i])**2
        i += 1
    mse = out/len(y_train)
    return mse

##Here we could get optimizer iterations and corresponding loss value
history3 = []
def callback3(x):
    fobj = f3(x)
    history3.append(fobj)

##Define the same functions for the validation data 
def f3_val(x):
    out = 0
    i = 0
    while i < len(y_test):
        out += (1/(1 + math.e**(-(x[0]*y_test[i] + x[1]))) - is_adult_test[i])**2
        i += 1
    mse = out/len(y_test)
    return mse

history3_val = []
def callback3_val(x):
    fobj = f3_val(x)
    history3_val.append(fobj)

##Iteration history for the training data
result = minimize(f3, [1,-150], method='Nelder-Mead', tol=1e-6, callback=callback3)
print(history3)

####Iteration history for the validation data
result_val = minimize(f3, [1,-150], method='Nelder-Mead', tol=1e-6, callback=callback3_val)
print(history3_val)

##Visualize training loss and validation loss
fig,ax = plt.subplots(1,1)
ax.plot(history3,'bo',label = 'training loss')
ax.plot(history3_val,'ro',label = 'validation loss')
ax.set_xlabel('optimizer iterations')
ax.set_ylabel('loss')
ax.legend()


#INITIAL GUESS 
#w = 1, b = -150
print("INITIAL GUESS: xo= 1,-150", " f3(xo)=",f3([1,-150]))
res = minimize(f3,[1,-150],method='Nelder-Mead', tol=1e-5,options={'disp': True} )
popt = res.x
print("OPTIMAL PARAM:",popt)
##OPTIMAL PARAM: OPTIMAL PARAM: [   6.17339692 -880.21306001]

##Plot training set, validation set and the fitted model 
fig,ax = plt.subplots(1,1)
ax.plot(y_train, is_adult, 'bo', label = 'training set')
ax.plot(y_test, is_adult_test,'go', label = 'validation set')
xl = np.linspace(0,250,100)
ax.plot(xl, 1/(1 + math.e**(-(xl*popt[0] + popt[1]))),'r',label = 'model')
ax.set_xlabel('weight(lb)')
ax.set_ylabel('is_adult')

###  END  ###











