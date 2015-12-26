# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 21:16:49 2015

@author: weizhi
"""


import pandas as pd

trainLabel = pd.read_csv('/Users/weizhi/Desktop/kaggle/whale detection/train 2.csv')
testLabel = pd.read_csv('/Users/weizhi/Desktop/kaggle/whale detection/sample_submission.csv')

import site; site.addsitedir("/usr/local/lib/python2.7/site-packages")
import cv2

img = cv2.imread('/Users/weizhi/Desktop/kaggle/whale detection/w_7489.jpg',-1)


#cv2.imshow('img',img)
#
#cv2.waitKey(0)
#cv2.destroyAllWindows()

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as nd

from skimage import data
from skimage.util import img_as_float
from skimage.filters import gabor_kernel
import cv2
#%% reading data
import matplotlib.cm as cm
import site; site.addsitedir("/usr/local/lib/python2.7/site-packages")
import numpy as np
from numpy import random
import cv2
import pylab as plt
# -*- coding: utf-8 -*-


#%% loading each files
import glob, os
import collections
import pandas as pd

# https://docs.python.org/2/library/os.html
def findFilePath(path):
    os.chdir(path)
    filePaths = []
    for file in glob.glob("*.jpg"):
        filePaths.append(file)
    return filePaths

#%% hours and generate the outputs
# deal with each csv file 
row = img.shape[0]
col = img.shape[1]
surf = cv2.SURF()
orb = cv2.ORB()
def readCSV(path, filePaths,row,col):
   # number = 500    
    data = np.zeros([100,100,len(filePaths)])
   # kpList = []
  #  desList = []    
    for i in range(len(filePaths)):
        print i
        filePath = path + '/' + filePaths[i]
        try:
            img1 =cv2.imread(filePath,0)
            img1 = cv2.resize(img1,(100,100))

        except:
            print "wrong image"

        data[:,:,i] = img1
    return data #,kpList,desList

def matchesGet(desList):
    matchesList = []
    for i in range(len(desList)-1):
        matches = bf.match(desList[i],desList[i+1])
        matches = sorted(matches, key = lambda x:x.distance)
        len(matches)
        matchesList.append(matches)

    return matchesList


#img2 = cv2.drawKeypoints(img1,kp1[:10],None,(255,0,0),4)
#plt.figure()
#plt.imshow(img2),plt.show()
#%% testing
path = '/Users/weizhi/Desktop/kaggle/whale detection/imgs/'
filePaths = findFilePath(path)


#%%
data  = readCSV(path , trainLabel['Image'],row,col)
testData = readCSV(path , testLabel['Image'],row,col)
target = trainLabel['whaleID']
#mathccesList = matchesGet(desList)


#%%
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
                 
                 input_shape=(None, 100*100),
                 dense0_num_units =2000,
                 
                 dropout_p=0.25,
                 
                 dense1_num_units = 2000,
                 dropout2_p=0.25,

                 dense2_num_units=1000,
             #    dropout3_p=0.25,


                 output_num_units=len(np.unique(yy)),
                 output_nonlinearity=softmax,
             #    output_nonlinearity=lasagne.nonlinearities.softmax,

                 update=nesterov_momentum,
                 update_learning_rate=0.00001,
                 update_momentum=0.9,
                 
                 eval_size=0.01,
                 verbose=1,
                 max_epochs=100) 




X = np.reshape(data,(data.shape[2],100*100))
from scipy import stats

col_mean = stats.nanmean(X,axis=0)
inds = np.where(np.isnan(X))

X[inds] = np.take(col_mean,inds[1])

from scipy import stats
#X =  stats.zscore(X)
#X2 = np.reshape(X,data.shape)
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(X)
X  = scaler.transform(X) 
cnn = net0.fit(X,yy) # train the CNN model for 15 epochs


#%%

X_test = np.reshape(testData,(testData.shape[2],100*100))
from scipy import stats

col_mean = stats.nanmean(X_test,axis=0)
inds = np.where(np.isnan(X_test))

X_test[inds] = np.take(col_mean,inds[1])
result = net0.predict_proba(X_test)
result_ = pd.DataFrame(result)

submit = pd.read_csv('/Users/weizhi/Desktop/kaggle/whale detection/sample_submission.csv')
submit.iloc[:,1:] = result
submit.to_csv('/Users/weizhi/Desktop/kaggle/whale detection/first_deeplearning.csv',index = False)


#%%
import lasagne

from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize
def CNN(n_epochs):
    net1 = NeuralNet(
        layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),      #Convolutional layer.  Params defined below
        ('pool1', layers.MaxPool2DLayer),   # Like downsampling, for execution speed
        ('conv2', layers.Conv2DLayer),
        ('hidden3', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],

    input_shape=(None, 1, 100, 100),
    conv1_num_filters=7, 
    conv1_filter_size=(3, 3), 
    conv1_nonlinearity=lasagne.nonlinearities.rectify,
        
    pool1_pool_size=(2, 2),
        
    conv2_num_filters=12, 
    conv2_filter_size=(2, 2),    
    conv2_nonlinearity=lasagne.nonlinearities.rectify,
        
    hidden3_num_units=1000,
    output_num_units=len(np.unique(yy)),
    output_nonlinearity=lasagne.nonlinearities.softmax,

    update_learning_rate=0.00001,
    update_momentum=0.9,
    
    max_epochs=n_epochs,
    verbose=1,
    )
    return net1
X = np.reshape(X,(data.shape[2],1,100,100))
cnn = CNN(100).fit(X,yy) # train the CNN model for 15 epochs


X_test = np.reshape(testData,(testData.shape[2],100*100))
from scipy import stats

col_mean = stats.nanmean(X_test,axis=0)
inds = np.where(np.isnan(X_test))

X_test[inds] = np.take(col_mean,inds[1])

X_test = np.reshape(X_test,(testData.shape[2],1,100,100))

result = cnn.predict_proba(X_test)
result_ = pd.DataFrame(result)

submit = pd.read_csv('/Users/weizhi/Desktop/kaggle/whale detection/sample_submission.csv')
submit.iloc[:,1:] = result
submit.to_csv('/Users/weizhi/Desktop/kaggle/whale detection/Second_deeplearning.csv',index = False)



#%%
net = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('hidden4', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 1, 28, 28),
    conv1_num_filters=1, conv1_filter_size=(3, 3), pool1_ds=(2, 2),
    conv2_num_filters=1, conv2_filter_size=(2, 2), pool2_ds=(2, 2),     
    hidden4_num_units=1000,
    output_num_units=len(np.unique(yy)),
    output_nonlinearity=lasagne.nonlinearities.softmax,

    update_learning_rate=0.01,
    update_momentum=0.9,

    regression=False,
    max_epochs=5,
    verbose=1)




















