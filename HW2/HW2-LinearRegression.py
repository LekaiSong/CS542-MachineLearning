#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame,Series
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression

#Converting numpy array to pandas dataframe
dataset = np.load('detroit.npy')
data = DataFrame({'FTP':dataset[:,0], 'UEMP':dataset[:,1], 'MAN':dataset[:,2], 'LIC':dataset[:,3], 'GR':dataset[:,4], 'NMAN':dataset[:,5], 'GOV':dataset[:,6], 'HE':dataset[:,7], 'WE':dataset[:,8], 'HOM':dataset[:,9]})
print(data)
#Finding out the potential variables
examDf = DataFrame(data)
new_examDf = examDf.ix[:,0:10]
print(new_examDf.describe())
print(new_examDf[new_examDf.isnull()==True].count())
print(new_examDf.corr()) #0-0.3 weak related; 0.3-0.6 average related; 0.6-1 strong related
sns.pairplot(data, x_vars=['FTP','HE','WE','NMAN','GOV'], y_vars='HOM',kind="reg", size=7, aspect=0.8)
plt.show() #Result shows FTP, HE, WE together are good predictors of HOM

#Splitting the dataset into the Training set and Test set
X_train,X_test,Y_train,Y_test = train_test_split(new_examDf[['FTP','HE','WE']],new_examDf.HOM,train_size=0.8)
print("independent variable:", new_examDf[['FTP','HE','WE']].shape, "；  training set:", X_train.shape, "；  test set:", X_test.shape)
print("controlled variable:", examDf.HOM.shape,"；  training set:", Y_train.shape, "；  test set:", Y_test.shape)
model = LinearRegression()
model.fit(X_train,Y_train) #linear regression
a = model.intercept_
b = model.coef_
print("intercept: ", a,",coef: ", b)

#display equation and correct to two decimal places
print("The best linear regression equation is: HOM =",round(a,2),"+",round(b[0],2),"* FTP +",round(b[1],2),"* HE +",round(b[2],2),"* WE")
 
Y_pred = model.predict(X_test) #use test set to predict
plt.plot(range(len(Y_pred)),Y_pred,'red', linewidth=2.5,label="predict data")
plt.plot(range(len(Y_test)),Y_test,'green',label="test data")
plt.legend(loc=2)
plt.show() #one of results: HOM = -79.01 + 0.35 *FTP + 6.14 *HE + -0.16 *WE