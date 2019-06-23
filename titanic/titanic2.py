#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
#pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns', None)
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from xgboost import XGBClassifier
#from lightgbm import LGBMClassifier
from sklearn import cross_validation, metrics
from sklearn.metrics import precision_score
from sklearn.ensemble import VotingClassifier
import warnings
warnings.filterwarnings('ignore')

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
#print(train_df['Name'])
combine_df = pd.concat([train_df,test_df])

#Title
#replace original titles with Mr, Mrs, and Miss
combine_df['Title'] = combine_df['Name'].apply(lambda x: x.split(', ')[1]).apply(lambda x: x.split('.')[0])
#print(combine_df['Title'])
combine_df['Title'] = combine_df['Title'].replace(['Sir'],'Mr')
combine_df['Title'] = combine_df['Title'].replace(['Mlle','Ms'],'Miss')
combine_df['Title'] = combine_df['Title'].replace(['Lady'],'Mrs')
combine_df['Title'] = combine_df['Title'].replace(['the Countess','Mme','Dona','Don','Major','Master','Capt','Jonkheer','Rev','Col','Dr'], 'Somebody')
#print(combine_df['Title'])
df = pd.get_dummies(combine_df['Title'],prefix='Title') #one-hot
combine_df = pd.concat([combine_df,df],axis=1)
#print(combine_df)

#Name_length
#name length distribution
combine_df['Name_Len'] = combine_df['Name'].apply(lambda x: len(x))
#print(combine_df['Name_Len'])
combine_df['Name_Len'] = pd.qcut(combine_df['Name_Len'],5)
#qcut depends on frequency, compared to cut.
#print(combine_df['Name_Len'])

#
##Dead_female_family & Survive_male_family
##combine_df['Surname'] = combine_df['Name'].apply(lambda x:x.split(',')[0])
#combine_df['Surname'] = combine_df['Name'].apply(lambda x:x.split(',')[1]).apply(lambda x: x.split('.')[1])
##print(combine_df['Surname'])
#dead_female_surname = list(set(combine_df[(combine_df.Sex=='female') & (combine_df.Age>=12)
#                              & (combine_df.Survived==0) & ((combine_df.Parch>0) | (combine_df.SibSp > 0))]['Surname'].values))
#survive_male_surname = list(set(combine_df[(combine_df.Sex=='male') & (combine_df.Age>=12)
#                              & (combine_df.Survived==1) & ((combine_df.Parch>0) | (combine_df.SibSp > 0))]['Surname'].values))
#combine_df['Dead_female_family'] = np.where(combine_df['Surname'].isin(dead_female_surname),0,1)
#combine_df['Survive_male_family'] = np.where(combine_df['Surname'].isin(survive_male_surname),0,1)
#combine_df = combine_df.drop(['Name','Surname'],axis=1)


#Age & isChild
group = combine_df.groupby(['Title', 'Pclass'])['Age']
combine_df['Age'] = group.transform(lambda x: x.fillna(x.median()))
combine_df = combine_df.drop('Title',axis=1)
#print(combine_df)
combine_df['IsChild'] = np.where(combine_df['Age']<=12,1,0)
#print(combine_df)
combine_df['Age'] = pd.cut(combine_df['Age'],5)
#print(combine_df)
#combine_df = combine_df.drop('Age',axis=1)
#print(combine_df)

##ticket
##print(combine_df['Ticket'])
#combine_df['Ticket_Lett'] = combine_df['Ticket'].apply(lambda x: str(x)[0])
#combine_df['Ticket_Lett'] = combine_df['Ticket_Lett'].apply(lambda x: str(x))
##print(combine_df['Ticket_Lett'])
#
#combine_df['High_Survival_Ticket'] = np.where(combine_df['Ticket_Lett'].isin(['1', '2', 'P']),1,0)
#combine_df['Low_Survival_Ticket'] = np.where(combine_df['Ticket_Lett'].isin(['A','W','3','7']),1,0)
#combine_df = combine_df.drop(['Ticket','Ticket_Lett'],axis=1)

#Embarked
#combine_df = combine_df.drop('Embarked',axis=1)
combine_df.Embarked = combine_df.Embarked.fillna('S')
df = pd.get_dummies(combine_df['Embarked'],prefix='Embarked')
combine_df = pd.concat([combine_df,df],axis=1).drop('Embarked',axis=1)

#FamilySize
combine_df['FamilySize'] = np.where(combine_df['SibSp']+combine_df['Parch']==0, 'Alone',
                                    np.where(combine_df['SibSp']+combine_df['Parch']<=3, 'Normal', 'Big'))
df = pd.get_dummies(combine_df['FamilySize'],prefix='FamilySize')
combine_df = pd.concat([combine_df,df],axis=1).drop(['SibSp','Parch','FamilySize'],axis=1)
#print(combine_df)


#Cabin
combine_df['Cabin_isNull'] = np.where(combine_df['Cabin'].isnull(),0,1)
combine_df = combine_df.drop('Cabin',axis=1)

#PClass
df = pd.get_dummies(combine_df['Pclass'],prefix='Pclass')
combine_df = pd.concat([combine_df,df],axis=1).drop('Pclass',axis=1)

#Sex
df = pd.get_dummies(combine_df['Sex'],prefix='Sex')
combine_df = pd.concat([combine_df,df],axis=1).drop('Sex',axis=1)

#Fare
combine_df['Fare'].fillna(combine_df['Fare'].dropna().median(),inplace=True)
#print(combine_df['Fare'])
combine_df['Low_Fare'] = np.where(combine_df['Fare']<=8.662,1,0)
combine_df['Normal_Fare'] = np.where((8.662<combine_df['Fare']) & (combine_df['Fare']<26),1,0)
combine_df['High_Fare'] = np.where(combine_df['Fare']>=26,1,0)
combine_df = combine_df.drop('Fare',axis=1)

print(combine_df.columns)
#print(combine_df)

features = combine_df.drop(["PassengerId","Survived"], axis=1).columns
#print(features)

#X = features
#y = combine_df['Survived']
#rf0 = RandomForestClassifier(oob_score=True, random_state=10)
#rf0.fit(X,y)
#print (rf0.oob_score_)
#y_predprob = rf0.predict_proba(X)[:,1]
#print ("AUC Score (Train): %f" % metrics.roc_auc_score(y, y_predprob))

LE = LabelEncoder()
for feature in features:
    LE = LE.fit(combine_df[feature])
    combine_df[feature] = LE.transform(combine_df[feature])
    
X_all = combine_df.iloc[:891,:].drop(["PassengerId","Survived"], axis=1)
Y_all = combine_df.iloc[:891,:]["Survived"]
X_test = combine_df.iloc[891:,:].drop(["PassengerId","Survived"], axis=1)

#tune model with hyperparameter
#knn - k
k_range = range(1,31)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_all.values, Y_all.values, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())
print(k_scores)
print("best k", k_scores.index(max(k_scores))+1) #result: k=11

##svc - c, kernel
#c_range = range(1,10)
#c_scores = []
#for c in c_range:
#    svc = SVC(C=c)
#    scores = cross_val_score(svc, X_all.values, Y_all.values, cv=10, scoring='accuracy')
#    c_scores.append(scores.mean())
#print(c_scores)
#print("best c", c_scores.index(max(c_scores))+1) #result: c=2 or 3
#
##kernels = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
#kernels_scores = []
#svc = SVC(C=2, kernel='linear')
#scores = cross_val_score(svc, X_all.values, Y_all.values, cv=10, scoring='accuracy')
#kernels_scores.append(scores.mean())
#svc = SVC(C=2, kernel='poly')
#scores = cross_val_score(svc, X_all.values, Y_all.values, cv=10, scoring='accuracy')
#kernels_scores.append(scores.mean())
#svc = SVC(C=2, kernel='rbf')
#scores = cross_val_score(svc, X_all.values, Y_all.values, cv=10, scoring='accuracy')
#kernels_scores.append(scores.mean())
#svc = SVC(C=2, kernel='sigmoid')
#scores = cross_val_score(svc, X_all.values, Y_all.values, cv=10, scoring='accuracy')
#kernels_scores.append(scores.mean())
##svc = SVC(C=2, kernel='precomputed') #cannot run
##scores = cross_val_score(svc, X_all.values, Y_all.values, cv=10, scoring='accuracy')
##kernels_scores.append(scores.mean())
#print(kernels_scores)
#print("best kernel", kernels_scores.index(max(kernels_scores))+1) #result: 'rbf', default
#
##LR - penalty
#l_scores = []
#LR = LogisticRegression(penalty='l1')
#scores = cross_val_score(LR, X_all.values, Y_all.values, cv=10, scoring='accuracy')
#l_scores.append(scores.mean())
#LR = LogisticRegression(penalty='l2')
#scores = cross_val_score(LR, X_all.values, Y_all.values, cv=10, scoring='accuracy')
#l_scores.append(scores.mean())
#print(l_scores)
#print("best penalty l", l_scores.index(max(l_scores))+1) #result: penalty='l1'

#RF
#n_range = range(100,500,30)
#n_scores = []
#for n in n_range:
#    RF = RandomForestClassifier(n_estimators=n,min_samples_leaf=4,class_weight={0:0.745,1:0.255})
#    scores = cross_val_score(RF, X_all.values, Y_all.values, cv=10, scoring='accuracy')
#    n_scores.append(scores.mean())
#print(n_scores)
#print("best n", (n_scores.index(max(n_scores))+1)*30) #result: n=~300

#param_test1 = {'n_estimators':range(50,500,20)}
#gsearch1 = GridSearchCV(estimator = RandomForestClassifier(min_samples_split=100, min_samples_leaf=20, max_depth=8,max_features='sqrt' ,random_state=10), 
#                       param_grid = param_test1, scoring='roc_auc',cv=5)
#gsearch1.fit(X_all.values,Y_all.values)
##print(gsearch1.grid_scores_)
#print("best n", gsearch1.best_params_, gsearch1.best_score_) #result: n=350
#
#param_test2 = {'max_depth':range(1,15), 'min_samples_split':range(30,151,10)}
#gsearch2 = GridSearchCV(estimator = RandomForestClassifier(n_estimators= 350, min_samples_leaf=20, max_features='sqrt' ,oob_score=True, random_state=10),
#                        param_grid = param_test2, scoring='roc_auc',iid=False, cv=5)
#gsearch2.fit(X_all.values,Y_all.values)
##print(gsearch2.grid_scores_)
#print("best max_depth and min_samples_split", gsearch2.best_params_, gsearch2.best_score_) #result: max_depth=5, min_samples_split=70

#param_test3 = {'min_samples_split':range(2,30), 'min_samples_leaf':range(1,5)}
#gsearch3 = GridSearchCV(estimator = RandomForestClassifier(n_estimators= 350, max_depth=5, max_features='sqrt' ,oob_score=True, random_state=10),
#                        param_grid = param_test3, scoring='roc_auc',iid=False, cv=5)
#gsearch3.fit(X_all.values,Y_all.values)
##print(gsearch3.grid_scores_)
#print("best min_samples_leaf and _split", gsearch3.best_params_, gsearch3.best_score_) #result: min_samples_leaf=1, min_samples_split=13

#param_test4 = {'max_features':range(1,20)}
#gsearch4 = GridSearchCV(estimator = RandomForestClassifier(n_estimators= 350, max_depth=5, min_samples_split=13, min_samples_leaf=1 ,oob_score=True, random_state=10),
#                        param_grid = param_test4, scoring='roc_auc',iid=False, cv=5)
#gsearch4.fit(X_all.values,Y_all.values)
##print(gsearch4.grid_scores_)
#print("best max_features", gsearch4.best_params_, gsearch4.best_score_) #result: max_features=18

##gbdt
#param_test1 = {'n_estimators':range(20,100,10)}
#gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=300, min_samples_leaf=20,max_depth=8,max_features='sqrt', subsample=0.8,random_state=10), 
#                        param_grid = param_test1, scoring='roc_auc',iid=False,cv=5)
#gsearch1.fit(X_all.values,Y_all.values)
##print(gsearch1.grid_scores_)
#print("best n", gsearch1.best_params_, gsearch1.best_score_) #result: n=70

#param_test2 = {'max_depth':range(1,15), 'min_samples_split':range(50,200,10)}
#gsearch2 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=70, min_samples_leaf=20, max_features='sqrt', subsample=0.8, random_state=10), 
#                        param_grid = param_test2, scoring='roc_auc',iid=False, cv=5)
#gsearch2.fit(X_all.values,Y_all.values)
##print(gsearch2.grid_scores_)
#print("best max_depth and min_samples_split", gsearch2.best_params_, gsearch2.best_score_) #result: max_depth=10, min_samples_split=100

#param_test3 = {'min_samples_split':range(10,150,10), 'min_samples_leaf':range(1,15)}
#gsearch3 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=70, max_depth=10, max_features='sqrt', subsample=0.8, random_state=10), 
#                        param_grid = param_test3, scoring='roc_auc',iid=False, cv=5)
#gsearch3.fit(X_all.values,Y_all.values)
##print(gsearch3.grid_scores_)
#print("best min_samples_leaf and _split", gsearch3.best_params_, gsearch3.best_score_) #result: min_samples_leaf=4, min_samples_split=120

#param_test4 = {'max_features':range(5,20)}
#gsearch4 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=70, max_depth=10, min_samples_leaf=4, min_samples_split=120, subsample=0.8, random_state=10), 
#                        param_grid = param_test4, scoring='roc_auc',iid=False, cv=5)
#gsearch4.fit(X_all.values,Y_all.values)
##print(gsearch4.grid_scores_)
#print("best max_features", gsearch4.best_params_, gsearch4.best_score_) #result: max_features=15

#param_test5 = {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]}
#gsearch5 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=70, max_depth=10, min_samples_leaf=4, min_samples_split=120, max_features=15, random_state=10), 
#                        param_grid = param_test5, scoring='roc_auc',iid=False, cv=5)
#gsearch5.fit(X_all.values,Y_all.values)
#print(gsearch5.grid_scores_)
#print("best subsample", gsearch5.best_params_, gsearch5.best_score_) #result: subsample=0.8


#optimize models with best parameters
LR = LogisticRegression(penalty='l1')
svc = SVC(C=2.0)
knn = KNeighborsClassifier(n_neighbors = 11)
DT = DecisionTreeClassifier()
RF = RandomForestClassifier(n_estimators=350,max_depth=5, min_samples_split=13,min_samples_leaf=1,max_features=18,oob_score=True,random_state=10)
gbdt = GradientBoostingClassifier(learning_rate=0.1, n_estimators=70, max_depth=10, min_samples_leaf=4, min_samples_split=120, max_features=15, random_state=10)
xgb = XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.03)
#lgb = LGBMClassifier(max_depth=3, n_estimators=500, learning_rate=0.02)
#clfs = [logreg, svc, knn, decision_tree, random_forest, gbdt, xgb, lgb]
classifiers = [LR, svc, knn, DT, RF, gbdt, xgb]

kfold = 10
cv_results = []
for classifier in classifiers:
    cv_results.append(cross_val_score(classifier, X_all.values, Y_all.values, scoring = "accuracy", cv = kfold, n_jobs=4))
#    print(cv_results)

cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

#ag = ["LR","SVC",'KNN','decision_tree',"random_forest","GBDT","xgbGBDT", "LGB"]
ag = ["LR","SVC",'KNN','DT',"RF","GBDT","xgbGBDT"]
cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":ag})

g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")
#plt.show()

for i in range(7):
    print(ag[i],cv_means[i])

#voting classifier
models = []
models.append(('xgb',xgb))
#models.append(('svc',svc))
models.append(('gbdt',gbdt))
models.append(('RF',RF))
models.append(('LR',LR))
#print(models)
ensemble_model = VotingClassifier(estimators=models)
scores = cross_val_score(ensemble_model, X_all.values, Y_all.values, cv=kfold)
print(scores.mean())

ensemble_model.fit(X_all.values, Y_all.values)
Y_test = ensemble_model.predict(X_test.values).astype(int)
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_test
    })
submission.to_csv(r'submission.csv', index=False)
