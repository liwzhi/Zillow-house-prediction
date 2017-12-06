# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 21:40:15 2015

@author: weizhi
"""

import pandas as pd

train = pd.read_csv('/Users/weizhi/Downloads/kaggle competion/train.csv')

sample_submit = pd.read_csv('/Users/weizhi/Downloads/kaggle competion/sample_submission.csv')
count =0
count1 = 0
X = train.iloc[:,:-1]
y = train.iloc[:,-1]
numberFeature = []
stringFeature1 = []
train = []
for types in X.dtypes:
    if types == 'int64' or types== 'float':
        numberFeature.append(count)
    if types =='object':
        stringFeature1.append(count)
    count +=1


numberFeatures =X.iloc[:,numberFeature]
#X = []
stringFeature = X.iloc[:,stringFeature1]
X = []
#%%
import pylab as plt
plt.figure()
plt.plot(numberFeatures.iloc[:,4])


def plotCorr(data,number):
    subData = data.iloc[number*1000:(number+1)*1000,:]
    corrY = subData.corr()
    plt.figure()
    plt.imshow(corrY)

#plotCorr(numberFeatures,1)
#plotCorr(numberFeatures,2)

#%%
from sklearn.feature_extraction import FeatureHasher
hasher = FeatureHasher(input_type='string',dtype='float')
strFeature = hasher.fit(stringFeature).transform(stringFeature).toarray()

#%% normalizetoin
numberFeatures = numberFeatures.fillna(0)

from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(numberFeatures.iloc[:,1:])
X_scaled = scaler.transform(numberFeatures.iloc[:,1:])
import numpy as np
#a = numpy.asarray([ [1,2,3], [4,5,6], [7,8,9] ])
#numpy.savetxt("/Users/weizhi/Downloads/kaggle competion/scared.csv", X_scaled, delimiter=",")

where_are_NaNs = np.isnan(X_scaled)
X_scaled[where_are_NaNs] = 0
#data = pd.read_csv("/Users/weizhi/Downloads/kaggle competion/scared.csv")
numberFeatures = None
import random 
index = random.sample(range(X_scaled.shape[0]),  50000)

trainSample = X_scaled[index,:]

target = y.iloc[index]

strSample = strFeature[:,index]

X = trainSample
#X= np.concatenate((trainSample,strSample.T),axis=1)
trainSample = None


#%% build the ROC feedback
from sklearn.metrics import roc_curve, auc , roc_auc_score
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=.3,
                                                    random_state=0)


X_train = np.log(1+X_train)
X_test = np.log(1+X_test)

where_are_NaNs = np.isnan(X_train)
X_train[where_are_NaNs] = 0


where_are_NaNs = np.isnan(X_test)
X_test[where_are_NaNs] = 0
#%%


from sklearn.ensemble import ExtraTreesClassifier

clf1 = ExtraTreesClassifier(n_estimators = 500, max_features = 'sqrt',max_depth=4,min_samples_split=4,random_state=369)
#clf1.fit(trainSample, target)

clf1.fit(X_train,y_train)
y_score = clf1.predict_proba(X_test)
fpr = dict()
tpr = dict()
roc_auc = dict()
score= roc_auc_score(y_test, y_score[:,1])
print ("the random forest score is %f" %score)



importances = clf1.feature_importances_
indices = np.argsort(importances)[::-1]

sumValue = 0
count = 0
for i in indices:
    count+=1
    if sumValue>0.8:
        break
    else:
        sumValue +=importances[i]
#        if count>=1881:
 #           print i


#plt.figure()
#plt.plot(importances[indices])

#%% train SVM 

from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4,1e-5],
                     'C': [ 10, 100, 1000]}]

#clf = GridSearchCV(SVC(C=1), tuned_parameters)
clf = SVC(kernel = 'rbf',gamma = 0.001,random_state=368,C=10,probability=True)
clf.fit(X_train[:,indices[:count]],y_train)

y_score = clf.predict_proba(X_test[:,indices[:count]])
score= roc_auc_score(y_test, y_score[:,1])
print ("the random forest score is %f" %score)

#clf.fit(trainSample[:,indices[:100]], target)

#%% xgboost

import xgboost as xgb

print("Train a XGBoost model")
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
import math
 
import numpy as np
 
 
 
class XGBoostClassifier():
    def __init__(self, num_boost_round=10, **params):
        self.clf = None
        self.num_boost_round = num_boost_round
        self.params = params
        self.params.update({'objective': 'multi:softprob'})
 
    def fit(self, X, y, num_boost_round=None):
        num_boost_round = num_boost_round or self.num_boost_round
        self.label2num = dict((label, i) for i, label in enumerate(sorted(set(y))))
        dtrain = xgb.DMatrix(X, label=[self.label2num[label] for label in y])
        self.clf = xgb.train(params=self.params, dtrain=dtrain, num_boost_round=num_boost_round)
 
    def predict(self, X):
        num2label = dict((i, label)for label, i in self.label2num.items())
        Y = self.predict_proba(X)
        y = np.argmax(Y, axis=1)
        return np.array([num2label[i] for i in y])
 
    def predict_proba(self, X):
        dtest = xgb.DMatrix(X)
        return self.clf.predict(dtest)
 
    def score(self, X, y):
        Y = self.predict_proba(X)
        return 1 / logloss(y, Y)
 
    def get_params(self, deep=True):
        return self.params
 
    def set_params(self, **params):
        if 'num_boost_round' in params:
            self.num_boost_round = params.pop('num_boost_round')
        if 'objective' in params:
            del params['objective']
        self.params.update(params)
        return self
    
    
def logloss(y_true, Y_pred):
    label2num = dict((name, i) for i, name in enumerate(sorted(set(y_true))))
    return -1 * sum(math.log(y[label2num[label]]) if y[label2num[label]] > 0 else -np.inf for y, label in zip(Y_pred, y_true)) / len(Y_pred)


xgb1 = XGBoostClassifier(
    eval_metric = 'auc',
    num_class = 2,
    nthread = 4,
    eta = 0.3,
    num_boost_round = 400,
    max_depth = 8,
    subsample = 0.9,
    colsample_bytree = 0.8,
    silent = 1,
    objective  = "binary:logistic",
    )
parameters = {
    'num_boost_round': [250,350, 500],
    'eta': [0.05, 0.1, 0.3],
    'max_depth': [6, 9, 12],
    'subsample': [0.9,0.8],
    'colsample_bytree': [0.7,0.9],
}


xgb1.fit(X_train[:,indices[:count]],y_train)

y_score = xgb1.predict_proba(X_test[:,indices[:count]])

score= roc_auc_score(y_test, y_score[:,1])
print ("the XGB forest score is %f" %score)
from sklearn.decomposition import PCA

from sklearn.decomposition import PCA, FastICA
pca = PCA(n_components=300)
pca.fit(X_train)


pcaFeature = pca.transform(X_train)
pcaTest = pca.transform(X_test)


XTrain= np.concatenate((pcaFeature,X_train[:,indices[:count]]),axis=1)

XTest= np.concatenate((pcaTest,X_test[:,indices[:count]]),axis=1)


xgb1.fit(XTrain,y_train)

y_score = xgb1.predict_proba(XTest)

score= roc_auc_score(y_test, y_score[:,1])
print ("the XGB forest score is %f" %score)








#from sklearn.manifold import TSNE
#model = TSNE(n_components=2, random_state=0)
#model.fit_transform(X_train[:,indices[:count]]) 

#%% KNN
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=7)

neigh.fit(X_train[:,indices[:count]],y_train)

y_score = neigh.predict_proba(X_test[:,indices[:count]])

score= roc_auc_score(y_test, y_score[:,1])
print ("the XGB forest score is %f" %score)
#%% reading the test data

# two steps, one normal and log and pca
test = pd.read_csv('/Users/weizhi/Downloads/kaggle competion/test.csv')
test = test.fillna(0)
testData = test.iloc[:,numberFeature]

stringFeatureNew = test.iloc[:,stringFeature1]


#%% value count and transform



#%%

#from sklearn.feature_extraction.text import CountVectorizer
#count= CountVectorizer()
#a = count.fit_transform(stringFeatureNew.iloc[:,1])
#
#from sklearn.feature_extraction import FeatureHasher
#hasher1 = FeatureHasher(input_type='string',dtype='float')
#strFeatureNew = hasher1.fit_transform(stringFeatureNew.T).toarray()





test = None


Test_scaled = scaler.transform(testData.iloc[:,1:])
where_are_NaNs = np.isnan(Test_scaled)
Test_scaled[where_are_NaNs] = 0
Test_scaled = np.log(1+Test_scaled)
testData =None
where_are_NaNs = np.isnan(Test_scaled)
Test_scaled[where_are_NaNs] = 0

#Test_scaled= np.concatenate((Test_scaled,strFeature),axis=1)



pcaTest = pca.transform(Test_scaled)


XTrain= np.concatenate((pcaTest,Test_scaled[:,indices[:count]]),axis=1)

#%% result
result = clf1.predict_proba(Test_scaled[:,indices[:count]])

result_svm = clf.predict_proba(Test_scaled[:,indices[:count]])

result_gbm = xgb1.predict(Test_scaled[:,indices[:count]])
#Test_scaled = None

#%% from kaggle
data = pd.read_csv('/Users/weizhi/Downloads/xgb3.csv')

#sample_submit['target'] = 0.1*result[:,1] + 0.65*data['target'] + 0.3*result_svm[:,1]
sample_submit['target']  = 1*result_gbm + 0.5*data['target'] + 0.5*result[:,1]

#sample_submit['target'] = result_svm[:,1]
sample_submit.to_csv('/Users/weizhi/Downloads/kaggle competion/sample_submission_gbm_PCA_400.csv',index=False)

