#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#process missing data
import pandas as pd
def impute_missing_data():
    data_training = pd.read_csv('crx.data.training', header=None, na_values='?')
    data_testing = pd.read_csv('crx.data.testing', header=None, na_values='?')
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
    X = train_data.iloc[:,:].values
    Y = test_data.iloc[:,:].values
    sc = StandardScaler()
    X[ : , [1,2,7,13,14]] = sc.fit_transform(X[ : , [1,2,7,13,14]]) #normalize only numerical features
    Y[ : , [1,2,7,13,14]] = sc.transform(Y[ : , [1,2,7,13,14]])
#    print(X)
#    print(Y)
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
        distances.append((i, distance(train_data[i], test_each))) #test_each is not a row here, but it is in predict()
        distances = sorted(distances, key=itemgetter(1)) #1 means column 1: 'distance()'
    return distances

#predict labels
def predict(train_data, test_data, k, which_set):
    dist = []
    neighbors = []
    closest = []
    predicts = []
    for test_each in test_data: #enumerate each row of test_data
        dist = distance_between(train_data, test_each)
        neighbors.append(dist) #all neighbors of each test row
#    return neighbors
    for neighbor in neighbors:
        close_i = []
        for i in range(k):
            close_i.append(neighbor[i])
        closest.append(close_i) #closest k neighbors of each test row [[()()()][()()()][()()()]...]
#    return closest
    for neighbor_array in closest: #closest k neighbors of first.. test row [()()()]
        if which_set == 1:
            one = two = three = 0
            for line_dist in neighbor_array: # line_dist is (row,dist) 
                if train_data[line_dist[0]][-1] == 3: three += 1
                elif train_data[line_dist[0]][-1] == 2: two += 1
                elif train_data[line_dist[0]][-1] == 1: one += 1
            if three >= two and three >= one: predicts.append(3)
            elif two >= three and two >= one: predicts.append(2)
            else: predicts.append(1)
        elif which_set == 2:
            plus = minus = 0
            for line_dist in neighbor_array: # (row,dist)
                if train_data[line_dist[0]][-1] == "+": # line_dist[0] is an exact row, train_data[line_dist[0]] is the row of original dataset, train_data[line_dist[0]][-1] is label column
                    plus += 1
                else:
                    minus += 1
            if plus > minus: # e.g. [(+)(+)(-)]
                predicts.append("+")
            else:
                predicts.append("-")
        else: 
            print("Error! Enter 1 or 2 please")
            break
    return predicts
    
def accuracy(labels_pred, test_data):
    labels_true = []
    correct = 0.0
    for each_row in test_data: # for [] in [[][][]..]
        labels_true.append(each_row[-1])
    for i in range(len(labels_pred)):
        if labels_pred[i] == labels_true[i]:
            correct += 1
    return float(correct/len(labels_pred))

if __name__ == "__main__":
    train_data, test_data = impute_missing_data()
    x, y = normalize_features(train_data, test_data)
    in_num = int(input("Which dataset you want to predict, 1:lenses, 2:crx? enter the number: "))
    if (in_num == 1):
        a = pd.read_csv('lenses.training').iloc[:,:].values
        b = pd.read_csv('lenses.testing').iloc[:,:].values
        print("The maximum of k is", len(a))
        k_num = int(input("What is k? enter a number: "))
        res1 = predict(a, b, k_num, in_num)
        print("accuracy = ", accuracy(res1, b), "when k = ", k_num, "and dataset is lenses")
    if (in_num == 2):
        print("The maximum of k is", len(x))
        k_num = int(input("What is k? enter a number: "))
        res2 = predict(x, y, k_num, in_num)
        print("accuracy = ", accuracy(res2, y), "when k = ", k_num, "and dataset is crx")


#Extra useful codes
##Handling the missing data
#from sklearn.preprocessing import Imputer
#imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
#imputer_x = imputer.fit(X[ : , [1,13]]) #column1,13 nan->mean
#imputer_y = imputer.fit(Y[ : , [1,13]])
#X[ : , [1,13]] = imputer_x.transform(X[ : , [1,13]])
#Y[ : , [1,13]] = imputer_y.transform(Y[ : , [1,13]])
#print(X)
#print(Y)