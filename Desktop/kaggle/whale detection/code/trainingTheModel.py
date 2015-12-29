# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 15:16:58 2015

@author: weizhi
"""

from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pandas as pd
from sklearn import grid_search
clf = SVC(probability=True,tol=0.01,gamma=0.1)


trainLabel = pd.read_csv('/Users/weizhi/Desktop/kaggle/whale detection/train 2.csv')

target = trainLabel['whaleID']

im_features = pd.read_csv('/Users/weizhi/Desktop/kaggle/whale detection/train_features.csv')


#%% cross validation
from sklearn import cross_validation
import xgboost as xgb
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
yy = le.fit_transform(target)

#%%
X_train, X_test, y_train, y_test = cross_validation.train_test_split(
     im_features, yy, test_size=0.01, random_state=0)
clf.fit(X_train, y_train)

#%% get the log loss
y_pred = clf.predict_proba(X_test)
from sklearn.metrics import log_loss
score = log_loss(y_test,y_pred)

#%% test feature 
#test_features
result = clf.predict_proba(test_features)

submit = pd.read_csv('/Users/weizhi/Desktop/kaggle/whale detection/sample_submission.csv')
submit.iloc[:,1:] = result
submit.to_csv('/Users/weizhi/Desktop/kaggle/whale detection/second_bagOfWords.csv',index = False)

training_names = '/Users/weizhi/Desktop/kaggle/whale detection/'
joblib.dump((clf, training_names, stdSlr, k, voc), training_names  +"/"+ "bof.pkl", compress=3)    
#%% training XGBoost

#X_train, X_valid,y_train,y_valid = train_test_split(X, yy, test_size=0.12, random_state=10)


dtrain = xgb.DMatrix(im_features, np.array(yy))
#dvalid = xgb.DMatrix(X_valid, y_valid)

#watchlist = [(dtrain, 'train'), (dvalid, 'eval')]

# train a XGBoost tree
print("Train a XGBoost model")
params = {"objective": "multi:softprob",
          "num_class":np.unique(yy).shape[0],
          "eta": 0.3,
          "max_depth": 10,
          "min_child_weight": 3,
          "silent": 1,
          "subsample": 0.95,
          "colsample_bytree": 0.7,
          "seed": 1}
num_trees=400
num_boost_round =1000,
gbm = xgb.train(params, dtrain, num_trees)
#%% get the result
result = gbm.predict(xgb.DMatrix(test_features))

submit = pd.read_csv('/Users/weizhi/Desktop/kaggle/whale detection/sample_submission.csv')
submit.iloc[:,1:] = result
submit.to_csv('/Users/weizhi/Desktop/kaggle/whale detection/xgboost.csv',index = False)

training_names = '/Users/weizhi/Desktop/kaggle/whale detection/'
joblib.dump((gbm, training_names, stdSlr, k, voc), training_names  +"/"+ "bof.pkl", compress=3)  

#%%training the model
import numpy as np
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder() 
yy = encoder.fit_transform(target.values).astype(np.int32)

from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
 
    
layers0 = [('input', InputLayer),
           ('dense0', DenseLayer),#100
           
           ('dropout', DropoutLayer),

           ('dense1', DenseLayer), # 200
           
          ('dropout2', DropoutLayer),

           ('dense2',DenseLayer), # 400



           ('output', DenseLayer)]
           
net0 = NeuralNet(layers=layers0,
                 
                 input_shape=(None, 300),
                 dense0_num_units =1000,
                 
                 dropout_p=0.25,
                 
                 dense1_num_units = 2000,
                 dropout2_p=0.25,

                 dense2_num_units=1000,
             #    dropout3_p=0.25,


                 output_num_units=len(np.unique(yy)),
                 output_nonlinearity=softmax,
             #    output_nonlinearity=lasagne.nonlinearities.softmax,

                 update=nesterov_momentum,
                 update_learning_rate=0.001,
                 update_momentum=0.9,
                 
                 eval_size=0.01,
                 verbose=1,
                 max_epochs=500) 





cnn = net0.fit(im_features,yy) # train the CNN model for 15 epochs






