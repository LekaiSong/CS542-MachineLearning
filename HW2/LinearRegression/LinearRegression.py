#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

#Define a basis function
def basisfunc(x):
    return x

#Import dataset
dataset = np.load('detroit.npy')
output_vector = [] #HOM y
for i in range(len(dataset)):
    output_vector.append(dataset[i][-1])
output = np.array(output_vector)

all_design_matrices = []
for k in range(1,8):
    design_matrix = []
    for i in range(len(dataset)):
        row = []
        row.append(basisfunc(dataset[i][0]))
        row.append(basisfunc(dataset[i][8]))
        row.append(basisfunc(dataset[i][k])) #two known varibles with an unknown varible
        design_matrix.append(row)
    all_design_matrices.append(design_matrix)
#print(all_design_matrices)

#Calculate error function
errors = []
for i in range(7): #7 combinations
    design = np.array(all_design_matrices[i])
    phi_sum = [0.0, 0.0, 0.0]
    tn_sum = sigma = loss =0.0
    for j in range(len(design)):
        phi_sum += basisfunc(design[j])
        tn_sum += basisfunc(output[j]) #output[n] is tn
    phi_mean = phi_sum/len(design) #Bishop 3.20
    tn_mean = tn_sum/len(output) #Bishop 3.20
#    weights = ((np.linalg.inv((design.T).dot(design))).dot(design.T)).dot(output) #Bishop 3.15
    weights = np.linalg.lstsq(design, output, rcond=-1) #equal to above
#    print("weights")
#    print(weights[0])
    for j in range(len(phi_mean)):
        sigma += weights[0][j]*phi_mean[j]
    w0 = tn_mean - sigma #Bishop 3.19
#    print("w0")
#    print(w0)
    for n in range(len(design)):
        sigma_w_phi = (weights[0]).dot(basisfunc(design[n]).T)
        loss += (output[n] - w0 - sigma_w_phi) ** 2 #Bishop 3.18
    Edw = loss*0.5
    errors.append(Edw)

print("Errors of UEMP, MAN, LIC, GR, NMAN, GOV, and HE: ", errors)
least_err = errors.index(min(errors))
if(least_err == 0): print("FTP, WE, and third variable UEMP determine HOM")
elif(least_err == 1): print("FTP, WE, and third variable MAN determine HOM")
elif(least_err == 2): print("FTP, WE, and third variable LIC determine HOM")
elif(least_err == 3): print("FTP, WE, and third variable GR determine HOM")
elif(least_err == 4): print("FTP, WE, and third variable NMAN determine HOM")
elif(least_err == 5): print("FTP, WE, and third variable GOV determine HOM")
elif(least_err == 6): print("FTP, WE, and third variable HE determine HOM")
print("------------------------------------------------------------------------")
print("------Code below is trying verifying the linear model with sklearn------")


#Code below is trying verifying the linear model
from pandas import DataFrame
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
#Converting numpy array to pandas dataframe
data = DataFrame({'FTP':dataset[:,0], 'UEMP':dataset[:,1], 'MAN':dataset[:,2], 'LIC':dataset[:,3], 'GR':dataset[:,4], 'NMAN':dataset[:,5], 'GOV':dataset[:,6], 'HE':dataset[:,7], 'WE':dataset[:,8], 'HOM':dataset[:,9]})
#print(data)

#Finding out the potential variables
examDf = DataFrame(data)
new_examDf = examDf.ix[:,0:10]
#print(new_examDf.describe())
#print(new_examDf[new_examDf.isnull()==True].count())
#print(new_examDf.corr()) #0-0.3 weak related; 0.3-0.6 average related; 0.6-1 strong related
sns.pairplot(data, x_vars=['FTP','WE', 'UEMP', 'MAN', 'LIC', 'GR', 'NMAN', 'GOV', 'HE'], y_vars='HOM',kind="reg", size=5, aspect=0.8)
plt.show()

#Splitting the dataset into the Training set and Test set
X_train,X_test,Y_train,Y_test = train_test_split(new_examDf[['FTP','WE','GR']],new_examDf.HOM,train_size=0.8) # section train and test set randomly
#print("independent variable:", new_examDf[['FTP','WE','GR']].shape, "；  training set:", X_train.shape, "；  test set:", X_test.shape)
#print("controlled variable:", examDf.HOM.shape,"；  training set:", Y_train.shape, "；  test set:", Y_test.shape)
model = LinearRegression()
model.fit(X_train,Y_train) #linear regression
a = model.intercept_
b = model.coef_
print("intercept: ", a,",coef: ", b)

#Display equation and correct to two decimal places
print("The best linear regression equation is: HOM =",round(a,2),"+",round(b[0],2),"* FTP +",round(b[1],2),"* WE +",round(b[2],2),"* GR")
 
Y_pred = model.predict(X_test) #use test set to predict
plt.plot(range(len(Y_pred)),Y_pred,'red', linewidth=2.5,label="predict data")
plt.plot(range(len(Y_test)),Y_test,'green',label="test data")
plt.legend(loc=2) #upper left
plt.show()