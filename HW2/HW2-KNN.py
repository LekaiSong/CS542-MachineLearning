#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#process missing data
import pandas as pd
def impute_missing_data():
    data_training = pd.read_csv('crx.data.training', header=None, na_values='?')
    data_testing = pd.read_csv('crx.data.testing', header=None, na_values='?')
#    data_describ = pd.read_csv('crx.names',delimiter="\t")
#    print(data_describ)
    data_training = data_training.fillna({0:'b'}) #column0 nan->b
    data_testing = data_testing.fillna({0:'b'})
    
    data_training[1] = data_training[1].fillna(data_training[1].mean()) #column1 nan->mean
    data_training[13] = data_training[13].fillna(data_training[13].mean()) #column13 nan->mean
    data_testing[1] = data_testing[1].fillna(data_testing[1].mean())
    data_testing[13] = data_testing[13].fillna(data_testing[13].mean())
    data_training = data_training.dropna(axis=0, how='any') #drop rows, but ids remain
    data_testing = data_testing.dropna(axis=0, how='any')
    pd.set_option('display.max_rows', None) #fully display
#    print(data_training)
#    print(data_testing)
    return data_training, data_testing

#normalize features
from sklearn.preprocessing import StandardScaler
def normalize_features(train_data, test_data):
#    X = data_training.iloc[:,:].values
    X = train_data.iloc[:,:].values
#    Y = data_testing.iloc[:,:].values
    Y = test_data.iloc[:,:].values
    sc = StandardScaler()
    X[ : , [1,2,7,13,14]] = sc.fit_transform(X[ : , [1,2,7,13,14]]) #normalize only numerical features
    Y[ : , [1,2,7,13,14]] = sc.transform(Y[ : , [1,2,7,13,14]])
    print(X)
    print(len(X))
    print(Y)
    return X, Y

#L2 distance
import math
def distance(row1, row2):
    dist = 0.0
    for i in range(len(row1) - 1): #not consider label column
        if type(row1[i]) is float:
            dist += (row1[i] - row2[i]) ** 2
        else:
            if row1[i] != row2[i]:
                dist += 1
    return math.sqrt(dist)

#caculate distances between training and testing data
from operator import itemgetter
def distance_between(train_data, test_each):
    distances = []
    for i in range(len(train_data)):
        distances.append((i, distance(train_data[i], test_each))) # test_each is not a row here, but it is in predict()
        distances = sorted(distances, key=itemgetter(1)) #1 means column 1: 'distance()'
    return distances

#predict
def predict(train_data, test_data, k):
    dist = []
    neighbors = []
    for test_each in test_data: # enumerate each row of test_data
        dist = distance_between(train_data, test_each)
        neighbors.append(dist)
    for n in neighbors:
            closei = []
            for i in range(k):
                closei.append(n[i])
            closest.append(closei)
        for neigh in closest:
            if filename == "lenses":
                three = 0
                two = 0
                one = 0
                for category in neigh:
                    if train_set[category[0]][-1] == "3":
                        three += 1
                    elif train_set[category[0]][-1] == "2":
                        two += 1
                    else:
                        one += 1
                if three > two and three > one:
                    predicts.append("3")
                elif two > three and two > one:
                    predicts.append("2")
                else:
                    predicts.append("1")
            else:
                plus = 0
                minus = 0
                for category in neigh:
                    if train_set[category[0]][-1] == "+":
                        plus += 1
                    else:
                        minus += 1
                if plus > minus:
                    predicts.append("+")
                else:
                    predicts.append("-")
        return predicts

if __name__ == "__main__":
    tra, tes = impute_missing_data()
    x, y = normalize_features(tra, tes)
#    print(distance_between(x, y))
    
##Handling the missing data
#from sklearn.preprocessing import Imputer
#imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
#imputer_x = imputer.fit(X[ : , [1,13]]) #column1,13 nan->mean
#imputer_y = imputer.fit(Y[ : , [1,13]])
#X[ : , [1,13]] = imputer_x.transform(X[ : , [1,13]])
#Y[ : , [1,13]] = imputer_y.transform(Y[ : , [1,13]])
#print(X)
#print(Y)