#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Numpy and Dataframe operations
import numpy as np
import pandas as pd
#pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns', None)

# Data visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Models
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
from xgboost import XGBClassifier

# Helper
from sklearn.preprocessing import LabelEncoder

# Cross-validation
from sklearn import cross_validation, metrics
from sklearn.metrics import precision_score
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
import warnings
warnings.filterwarnings('ignore')

#Load data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
combine_df = pd.concat([train_df,test_df])
print("--------------------------------------------------------")
print("Data Cleaning ...")

#Title
#replace original titles with Mr, Mrs, Miss, Royalty and Officer
combine_df['Title'] = combine_df['Name'].apply(lambda x: x.split(', ')[1]).apply(lambda x: x.split('.')[0])
#print(combine_df['Title'])
combine_df['Title'] = combine_df['Title'].replace(['Mr'],'Mr')
combine_df['Title'] = combine_df['Title'].replace(['Mlle','Miss'],'Miss')
combine_df['Title'] = combine_df['Title'].replace(['Mme','Ms'],'Mrs')
combine_df['Title'] = combine_df['Title'].replace(['the Countess','Dona','Don','Sir','Lady','Jonkheer'], 'Royalty')
combine_df['Title'] = combine_df['Title'].replace(['Capt','Col','Major','Dr','Rev'],'Officer')
#print(combine_df['Title'])
df = pd.get_dummies(combine_df['Title'],prefix='Title') #one-hot
combine_df = pd.concat([combine_df,df],axis=1)
#print(combine_df)

#Name_length
combine_df['Name_Len'] = combine_df['Name'].apply(lambda x: len(x))
#print(combine_df['Name_Len'])
combine_df['Name_Len'] = pd.qcut(combine_df['Name_Len'],5)
#qcut depends on frequency, compared to cut.
#print(combine_df['Name_Len'])

#Family survival
combine_df['Surname'] = combine_df['Name'].apply(lambda x: str.split(x, ".")[1].split()[0])
#print(combine_df['Surname'])
combine_df['Fare'].fillna(combine_df['Fare'].mean(), inplace=True)

DEFAULT_SURVIVAL_VALUE = 0.5
combine_df['Family_Survival'] = DEFAULT_SURVIVAL_VALUE

for grp, grp_df in combine_df[['Survived','Name', 'Surname', 'Fare', 'Ticket', 'PassengerId', 'SibSp', 'Parch', 'Age', 'Cabin']].groupby(['Surname', 'Fare']):
    if (len(grp_df) != 1):
        # A Family group is found.
        for index, row in grp_df.iterrows():
            smax = grp_df.drop(index)['Survived'].max()
            smin = grp_df.drop(index)['Survived'].min()
            passID = row['PassengerId']
            if (smax == 1.0):
                combine_df.loc[combine_df['PassengerId'] == passID, 'Family_Survival'] = 1
            elif (smin==0.0):
                combine_df.loc[combine_df['PassengerId'] == passID, 'Family_Survival'] = 0
#print("Number of passengers with family survival information:", combine_df.loc[combine_df['Family_Survival']!=0.5].shape[0])

for _, grp_df in combine_df.groupby('Ticket'):
    if (len(grp_df) != 1):
        for index, row in grp_df.iterrows():
            if (row['Family_Survival'] == 0) | (row['Family_Survival']== 0.5):
                smax = grp_df.drop(index)['Survived'].max()
                smin = grp_df.drop(index)['Survived'].min()
                passID = row['PassengerId']
                if (smax == 1.0):
                    combine_df.loc[combine_df['PassengerId'] == passID, 'Family_Survival'] = 1
                elif (smin==0.0):
                    combine_df.loc[combine_df['PassengerId'] == passID, 'Family_Survival'] = 0                      
#print("Number of passenger with family/group survival information: " +str(combine_df[combine_df['Family_Survival']!=0.5].shape[0]))
combine_df = combine_df.drop(['Name', 'Surname'],axis=1)

#Age & isChild
group = combine_df.groupby(['Title', 'Pclass'])['Age']
combine_df['Age'] = group.transform(lambda x: x.fillna(x.median()))
combine_df = combine_df.drop('Title',axis=1)
#print(combine_df)
combine_df['IsChild'] = np.where(combine_df['Age']<=12,1,0)
combine_df['AgeBin'] = pd.qcut(combine_df['Age'],5)
combine_df = combine_df.drop('Age',axis=1)
#print(combine_df)

#ticket
#print(combine_df['Ticket'])
combine_df['Ticket_Lett'] = combine_df['Ticket'].apply(lambda x: str(x)[0])
combine_df['Ticket_Lett'] = combine_df['Ticket_Lett'].apply(lambda x: str(x))
#print(combine_df['Ticket_Lett'])
combine_df['High_Survival_Ticket'] = np.where(combine_df['Ticket_Lett'].isin(['1','2','P']),1,0)
combine_df['Low_Survival_Ticket'] = np.where(combine_df['Ticket_Lett'].isin(['3','7','A','W']),1,0)
combine_df = combine_df.drop(['Ticket','Ticket_Lett'],axis=1)

#Embarked
#combine_df = combine_df.drop('Embarked',axis=1)
combine_df.Embarked = combine_df.Embarked.fillna('S')
df = pd.get_dummies(combine_df['Embarked'],prefix='Embarked')
combine_df = pd.concat([combine_df,df],axis=1).drop('Embarked',axis=1)

#FamilySize
combine_df['FamilySize'] = combine_df['SibSp']+combine_df['Parch']
combine_df = combine_df.drop(['SibSp','Parch'],axis=1)
#combine_df['FamilySize'] = np.where(combine_df['SibSp']+combine_df['Parch']==0, 'Alone',
#                                    np.where(combine_df['SibSp']+combine_df['Parch']<=3, 'Normal', 'Big'))
#df = pd.get_dummies(combine_df['FamilySize'],prefix='FamilySize')
#combine_df = pd.concat([combine_df,df],axis=1).drop(['SibSp','Parch','FamilySize'],axis=1)
#print(combine_df)

#Cabin
combine_df['Cabin_isNull'] = np.where(combine_df['Cabin'].isnull(),0,1)
combine_df = combine_df.drop('Cabin',axis=1)

#PClass
#df = pd.get_dummies(combine_df['Pclass'],prefix='Pclass')
#combine_df = pd.concat([combine_df,df],axis=1).drop('Pclass',axis=1)

#Sex
combine_df['Sex'].replace(['male','female'],[0,1],inplace=True)
#df = pd.get_dummies(combine_df['Sex'],prefix='Sex')
#combine_df = pd.concat([combine_df,df],axis=1).drop('Sex',axis=1)

#Fare
#combine_df['Fare'].fillna(combine_df['Fare'].dropna().median(),inplace=True)
combine_df['FareBin'] = pd.qcut(combine_df['Fare'],5)
combine_df = combine_df.drop('Fare',axis=1)
#print(combine_df['Fare'])
#combine_df['Low_Fare'] = np.where(combine_df['Fare']<=8.662,1,0)
#combine_df['Normal_Fare'] = np.where((8.662<combine_df['Fare']) & (combine_df['Fare']<26),1,0)
#combine_df['High_Fare'] = np.where(combine_df['Fare']>=26,1,0)
#combine_df = combine_df.drop('Fare',axis=1)

#print(combine_df.columns)
#print(combine_df)

features = combine_df.drop(["PassengerId","Survived"], axis=1).columns
#print(features)

LE = LabelEncoder()
for feature in features:
    LE = LE.fit(combine_df[feature])
    combine_df[feature] = LE.transform(combine_df[feature])
print("Data Cleaning done")

X_all = combine_df.iloc[:891,:].drop(["PassengerId","Survived"], axis=1)
Y_all = combine_df.iloc[:891,:]["Survived"]
X_test = combine_df.iloc[891:,:].drop(["PassengerId","Survived"], axis=1)
print("--------------------------------------------------------")
print("Features before selection: ")
print(X_all.columns)
print("--------------------------------------------------------")

##Scaling features
print("Features Scaling ...")
X_all.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
X_test.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
print("Features Scaling done")
print("--------------------------------------------------------")

#features selection
print("Features Importance with RandomForest")
clf = RandomForestClassifier(n_estimators=350, oob_score=True)
clf = clf.fit(X_all.values, Y_all.values)
features = pd.DataFrame()
features['feature'] = X_all.columns
features['importance'] = clf.feature_importances_
features.sort_values(by=['importance'], ascending=True, inplace=True)
features.set_index('feature', inplace=True)
features.plot(kind='barh', figsize=(15,8))
plt.show()
#delete useless features
X_all = X_all.drop(["Title_Royalty","Title_Officer","Title_Mrs","Title_Miss","Title_Master",
                    "Embarked_Q","Embarked_C","Embarked_S","IsChild","Cabin_isNull",
                    "High_Survival_Ticket","Low_Survival_Ticket"],axis=1)
X_test = X_test.drop(["Title_Royalty","Title_Officer","Title_Mrs","Title_Miss","Title_Master",
                      "Embarked_Q","Embarked_C","Embarked_S","IsChild","Cabin_isNull",
                      "High_Survival_Ticket","Low_Survival_Ticket"],axis=1)
print("--------------------------------------------------------")
print("Features after selection: ")
print(X_all.columns)
print("--------------------------------------------------------")

'''
#tune model with hyperparameter
#You dont need to tune the hyperparameters every time actually, unless you modified the Feature Enginnering above.

#knn - k
k_range = range(1,31)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_all.values, Y_all.values, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())
print(k_scores)
print("For knn, the best k is", k_scores.index(max(k_scores))+1) #result: k=17
print("--------------------------------------------------------")

#svc - c, kernel
c_range = range(1,20)
c_scores = []
for c in c_range:
    svc = SVC(C=c)
    scores = cross_val_score(svc, X_all.values, Y_all.values, cv=10, scoring='accuracy')
    c_scores.append(scores.mean())
print(c_scores)
print("For svc, the best c is", c_scores.index(max(c_scores))+1) #result: c=3
print("--------------------------------------------------------")

#kernels = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
kernels_scores = []
svc = SVC(C=2, kernel='linear')
scores = cross_val_score(svc, X_all.values, Y_all.values, cv=10, scoring='accuracy')
kernels_scores.append(scores.mean())
svc = SVC(C=2, kernel='poly')
scores = cross_val_score(svc, X_all.values, Y_all.values, cv=10, scoring='accuracy')
kernels_scores.append(scores.mean())
svc = SVC(C=2, kernel='rbf')
scores = cross_val_score(svc, X_all.values, Y_all.values, cv=10, scoring='accuracy')
kernels_scores.append(scores.mean())
svc = SVC(C=2, kernel='sigmoid')
scores = cross_val_score(svc, X_all.values, Y_all.values, cv=10, scoring='accuracy')
kernels_scores.append(scores.mean())
#svc = SVC(C=2, kernel='precomputed') #cannot run
#scores = cross_val_score(svc, X_all.values, Y_all.values, cv=10, scoring='accuracy')
#kernels_scores.append(scores.mean())
print(kernels_scores)
print("For svc, the best kernel is", kernels_scores.index(max(kernels_scores))+1) #result: 'poly'
print("--------------------------------------------------------")

#LR - penalty
l_scores = []
LR = LogisticRegression(penalty='l1')
scores = cross_val_score(LR, X_all.values, Y_all.values, cv=10, scoring='accuracy')
l_scores.append(scores.mean())
LR = LogisticRegression(penalty='l2')
scores = cross_val_score(LR, X_all.values, Y_all.values, cv=10, scoring='accuracy')
l_scores.append(scores.mean())
print(l_scores)
print("For LR, the best penalty is L", l_scores.index(max(l_scores))+1) #result: penalty='l1'
print("--------------------------------------------------------")

#RF
param_test1 = {'n_estimators':range(50,500,20)}
gsearch1 = GridSearchCV(estimator = RandomForestClassifier(min_samples_split=100, min_samples_leaf=20, max_depth=8, max_features='sqrt', random_state=10), 
                       param_grid = param_test1, scoring='roc_auc',cv=5)
gsearch1.fit(X_all.values,Y_all.values)
#print(gsearch1.grid_scores_)
print("For RF, the best n is", gsearch1.best_params_, "score is", gsearch1.best_score_) #result: n=410
print("--------------------------------------------------------")

param_test2 = {'max_depth':range(1,15), 'min_samples_split':range(10,200,10)}
gsearch2 = GridSearchCV(estimator = RandomForestClassifier(n_estimators=410, min_samples_leaf=20, max_features='sqrt', oob_score=True, random_state=10),
                        param_grid = param_test2, scoring='roc_auc',iid=False, cv=5)
gsearch2.fit(X_all.values,Y_all.values)
#print(gsearch2.grid_scores_)
print("For RF, the best max_depth and min_samples_split are", gsearch2.best_params_, "score is", gsearch2.best_score_) #result: max_depth=6, min_samples_split=30(double check below)
print("--------------------------------------------------------")

param_test3 = {'min_samples_split':range(2,30), 'min_samples_leaf':range(1,5)}
gsearch3 = GridSearchCV(estimator = RandomForestClassifier(n_estimators=410, max_depth=6, max_features='sqrt', oob_score=True, random_state=10),
                        param_grid = param_test3, scoring='roc_auc',iid=False, cv=5)
gsearch3.fit(X_all.values,Y_all.values)
#print(gsearch3.grid_scores_)
print("For RF, the best min_samples_leaf and _split are", gsearch3.best_params_, "score is", gsearch3.best_score_) #result: min_samples_leaf=1, min_samples_split=4
print("--------------------------------------------------------")

param_test4 = {'max_features':range(1,5)}
gsearch4 = GridSearchCV(estimator = RandomForestClassifier(n_estimators=410, max_depth=6, min_samples_split=4, min_samples_leaf=1, oob_score=True, random_state=10),
                        param_grid = param_test4, scoring='roc_auc',iid=False, cv=5)
gsearch4.fit(X_all.values,Y_all.values)
#print(gsearch4.grid_scores_)
print("For RF, the best max_features is", gsearch4.best_params_, "score is", gsearch4.best_score_) #result: max_features=4
print("--------------------------------------------------------")

#gbdt
param_test1 = {'n_estimators':range(80,200,10)}
gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=300, min_samples_leaf=20, max_depth=8, max_features='sqrt', subsample=0.8, random_state=10), 
                        param_grid = param_test1, scoring='roc_auc',iid=False,cv=5)
gsearch1.fit(X_all.values,Y_all.values)
#print(gsearch1.grid_scores_)
print("For gbdt, the best n is", gsearch1.best_params_, "score is", gsearch1.best_score_) #result: n=160
print("--------------------------------------------------------")

param_test2 = {'max_depth':range(1,15), 'min_samples_split':range(100,300,20)}
gsearch2 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=160, min_samples_leaf=20, max_features='sqrt', subsample=0.8, random_state=10), 
                        param_grid = param_test2, scoring='roc_auc',iid=False, cv=5)
gsearch2.fit(X_all.values,Y_all.values)
#print(gsearch2.grid_scores_)
print("For gbdt, the best max_depth and min_samples_split are", gsearch2.best_params_, "score is", gsearch2.best_score_) #result: max_depth=5, min_samples_split=260(double check below)
print("--------------------------------------------------------")

param_test3 = {'min_samples_split':range(10,150,10), 'min_samples_leaf':range(1,15)}
gsearch3 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=160, max_depth=5, max_features='sqrt', subsample=0.8, random_state=10), 
                        param_grid = param_test3, scoring='roc_auc',iid=False, cv=5)
gsearch3.fit(X_all.values,Y_all.values)
#print(gsearch3.grid_scores_)
print("For gbdt, the best min_samples_leaf and _split are", gsearch3.best_params_, "score is", gsearch3.best_score_) #result: min_samples_leaf=3, min_samples_split=130
print("--------------------------------------------------------")

param_test4 = {'max_features':range(1,5)}
gsearch4 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=160, max_depth=5, min_samples_leaf=3, min_samples_split=130, subsample=0.8, random_state=10), 
                        param_grid = param_test4, scoring='roc_auc',iid=False, cv=5)
gsearch4.fit(X_all.values,Y_all.values)
#print(gsearch4.grid_scores_)
print("For gbdt, the best max_features is", gsearch4.best_params_, "score is", gsearch4.best_score_) #result: max_features=2
print("--------------------------------------------------------")

param_test5 = {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]}
gsearch5 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=160, max_depth=5, min_samples_leaf=3, min_samples_split=130, max_features=2, random_state=10), 
                        param_grid = param_test5, scoring='roc_auc',iid=False, cv=5)
gsearch5.fit(X_all.values,Y_all.values)
#print(gsearch5.grid_scores_)
print("For gbdt, the best subsample is", gsearch5.best_params_, "score is", gsearch5.best_score_) #result: subsample=0.8
print("--------------------------------------------------------")

#decision tree
param_test1 = {'criterion': ['gini', 'entropy'], 'max_depth': range(1,15), 'min_samples_split': range(2,20),
              'min_samples_leaf': range(1,10), 'max_features': range(1,5)}
gsearch1 = GridSearchCV(estimator = DecisionTreeClassifier(random_state=10), 
                        param_grid = param_test1, scoring='roc_auc',iid=False,cv=5)
gsearch1.fit(X_all.values,Y_all.values)
#print(gsearch1.grid_scores_)
print("For DT, the best parameters are", gsearch1.best_params_, "score is", gsearch1.best_score_) #result: criterion='gini', max_depth=11, max_features=4, min_samples_leaf=2, min_samples_split=14
print("--------------------------------------------------------")

#xgb
param_test1 = {'max_depth': range(1,10), 'n_estimators': range(50,100,10), 'learning_rate': [0.01, 0.03, 0.1, 0.2]}
gsearch1 = GridSearchCV(estimator = XGBClassifier(random_state=10), 
                        param_grid = param_test1, scoring='roc_auc',iid=False,cv=5)
gsearch1.fit(X_all.values,Y_all.values)
#print(gsearch1.grid_scores_)
print("For xgb, the best parameters are", gsearch1.best_params_, "score is", gsearch1.best_score_) #result: learning_rate=0.1, max_depth=3, n_estimators=70
print("--------------------------------------------------------")
'''

#Optimize models, best parameters above maybe not also best for test set. Thus optimize slightly according to result.  
LR = LogisticRegression(penalty='l1')
svc = SVC(C=3.0, kernel='poly')
knn = KNeighborsClassifier(n_neighbors = 17)
DT = DecisionTreeClassifier(criterion='gini', max_depth=11, max_features=4, min_samples_leaf=2, min_samples_split=14, random_state=10)
RF = RandomForestClassifier(n_estimators=410,max_depth=6, min_samples_split=4,min_samples_leaf=1,max_features=4,oob_score=True,random_state=10)
gbdt = GradientBoostingClassifier(learning_rate=0.1, n_estimators=160, max_depth=5, min_samples_leaf=3, min_samples_split=130, max_features=2, subsample=0.8, random_state=10)
xgb = XGBClassifier(max_depth=3, n_estimators=70, learning_rate=0.1)
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

ag = ["LR","SVC",'KNN','DT',"RF","GBDT","xgbGBDT"]
cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":ag})

g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")
print("Cross validation scores of models")
plt.show()
print("--------------------------------------------------------")

print("Scores of models:")
for i in range(7):
    print(ag[i],cv_means[i])

#voting classifier
models = []
models.append(('xgb',xgb))
models.append(('svc',svc))
models.append(('gbdt',gbdt))
models.append(('RF',RF))
models.append(('LR',LR))
models.append(('DT',DT))
models.append(('knn',knn))
#print(models)
ensemble_model = VotingClassifier(estimators=models)
scores = cross_val_score(ensemble_model, X_all.values, Y_all.values, cv=kfold)
print("--------------------------------------------------------")
print("Ensemble model score:")
print(scores.mean())
print("--------------------------------------------------------")

ensemble_model.fit(X_all.values, Y_all.values)
Y_test = ensemble_model.predict(X_test.values).astype(int)
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_test
    })
submission.to_csv(r'submission.csv', index=False)
print("submission.csv is already created")
print("--------------------------------------------------------")