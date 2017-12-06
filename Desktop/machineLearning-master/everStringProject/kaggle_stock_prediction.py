# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 16:05:54 2015

@author: weizhi


train.csv - the training set, including the columns of:
Feature_1 - Feature_25
Ret_MinusTwo, Ret_MinusOne
Ret_2 - Ret_120
Ret_121 - Ret_180: target variables
Ret_PlusOne, Ret_PlusTwo: target variables
Weight_Intraday, Weight_Daily


test.csv - the test set, including the columns of:
Feature_1 - Feature_25
Ret_MinusTwo, Ret_MinusOne
Ret_2 - Ret_120
sample_submission.csv - a sample submission file in the correct format

"""
import numpy as np
import pandas as pd

train = pd.read_csv('/Users/weizhi/Desktop/kaggle stock data/train.csv');

test = pd.read_csv('/Users/weizhi/Desktop/kaggle stock data/test.csv');
submit = pd.read_csv('/Users/weizhi/Desktop/kaggle stock data/sample_submission.csv');

for i in range(665,666):
    train.iloc[i,28:147].plot()
    
    
#%% build the classfiation probelms. target is from 

X = train.iloc[:,26:147].fillna(train.iloc[:,26:147].mean())    # get the features 
y = train.iloc[:,147:209].fillna(train.iloc[:,147:209].mean())   # get teh target 

#train = []
#%% for prediction 
testData = test.iloc[:,26:149].fillna(test.iloc[:,26:149].mean())

#test = []
#%% build the classification 
from sklearn import linear_model

#regr = linear_model.LinearRegression()
regr = linear_model.Lasso(alpha=0.4)
# Train the model using the training sets
regr.fit(X, y)

res = regr.predict(testData)

result = np.reshape(res,submit.shape[0])

#%% feature extraction from regression data sets
import xgboost as xgb

#rf.fit(train[features],train["signal"])

print("Train a XGBoost model")
params = {"objective": "reg:linear",
          "learning_rate": 0.2,
          "max_depth": 6,
          "min_child_weight": 3,
          "silent": 1,
          "subsample": 0.7,
          "colsample_bytree": 0.7,
          "seed": 1}
          
num_trees=50

#clf = GridSearchCV(xgb_model, params, n_jobs=4, 
 #                  cv=StratifiedKFold(train['signal'], n_folds=5, shuffle=True), 
 #                  verbose=2, refit=True)

#clf.fit(train[features], train["signal"])

#best_parameters, score, _ = max(clf.grid_scores_, key=lambda x: x[1])
#print('Raw AUC score:', score)
#for param_name in sorted(best_parameters.keys()):
#    print("%s: %r" % (param_name, best_parameters[param_name]))

resultSave = np.zeros([60000,62])
for i in range(62):
    print "the training data loop is"
    print i 
    target = y.iloc[:,i]
    gbm = xgb.train(params, xgb.DMatrix(X, target), num_trees)

    currResult = gbm.predict(xgb.DMatrix(testData))
    resultSave[:,i] = currResult
    # update the features
    X = pd.concat([X,target],axis=1)
    testData = pd.concat([testData,pd.DataFrame(currResult)],axis=1)
    
    
    
    
#%% make the submission 
result = np.reshape(resultSave,[60000*62,1],axis=0)
submit['Predicted'] = result

submit.to_csv('/Users/weizhi/Desktop/kaggle stock data/fourth_submission.csv',index = False)





