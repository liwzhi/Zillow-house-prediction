# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 23:50:05 2015

@author: Algorithm 001
"""


import glob, os
import numpy as np
import random 
import pandas as pd

# https://docs.python.org/2/library/os.html
def findFilePath(path):
    """
    Input: raw CSV data path
    Output: list of file path
    
    """
    os.chdir(path)
    filePaths = []
    for file in glob.glob("*.csv"):
        filePaths.append(file)
    return filePaths
    
def readCSV(filePath):
        data = pd.read_csv(filePath)
        data.columns = range(1,9)
        return data  
    

Path = '/Users/weizhi/Downloads/TrainingData'   
#Path = os.chdir('/Users/weizhi/Downloads/TrainingData')
#%%
Dict = {}
Dict['1B'] = ['C:/Users/Algorithm 001/Desktop/Drawing files/1B/3-11-13_09.59.57/']
Dict['1B'].append('C:/Users/Algorithm 001/Desktop/Drawing files/1B/3-11-13_09.59.28/')

B1file=  findFilePath(Dict['1B'][1])

healthPath = []
particial = []
hype = []

for item in B1file:
    if item.find('health')>0:
        healthPath.append(item)
    if item.find('Less')>0:
        particial.append(item)
    if item.find('hyperemia')>0:
        hype.append(item)


healthData = readCSV(healthPath[1])
particialData = readCSV(particial[1])
hype = readCSV(hype[1])


#%%%%%%%
Dict['1B1hr'] = ['C:/Users/Algorithm 001/Desktop/Drawing files/1B1hr/2013-11-13_10.48.24/']
Dict['1B1hr'].append('C:/Users/Algorithm 001/Desktop/Drawing files/1B1hr/2013-11-13_10.48.50/')
 
B1file1=  findFilePath(Dict['1B1hr'][1])

healthPath1 = []
particial1 = []
hype1 = []

for item in B1file1:
    if item.find('health')>0:
        healthPath1.append(item)
    if item.find('Less')>0:
        particial1.append(item)
    if item.find('hyperemia')>0:
        hype1.append(item)


healthData1 = readCSV(healthPath1[1])
particialData1 = readCSV(particial1[1])
hype1 = readCSV(hype1[1])

#%% concate the data for the training sets
def randomSample(df,df1,label):
    rows = random.sample(df.index,2000)
    df = df.ix[rows]
    rows1 = random.sample(df1.index,2000)
    df3 = df1.ix[rows1]
    df2 = pd.concat([df,df3],axis=0)
    df2 = df2.reset_index()
    del df2['index']
    labels = pd.DataFrame()
    labels['label'] = [label]*4000
    df2 = pd.concat([df2,labels],axis=1)

    return df2

healthTrain = randomSample(healthData,healthData1,1)
particialTrain = randomSample(particialData,particialData1,2)
hyperTrain = randomSample(hype,hype1,3)

B1_particial = particialTrain

#%%

Data = pd.concat([healthTrain,hyperTrain,particialTrain],axis=0)
Data = Data.reset_index()
#print Data.tail(4)

df = Data.iloc[np.random.permutation(len(Data))]
del df['index']

#%%
col = Data.columns
X = df.loc[:,col[1:-1]]
y = df.loc[:,col[-1]]


#%% go the the modeling part
from sklearn.lda import LDA
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
clf = LDA()

X_train, X_test, y_train, y_test = cross_validation.train_test_split( X,y, test_size=0.5, random_state=0)

y_pred = clf.fit(X_train, y_train).predict(X_test)
clf = clf.fit(X_train,y_train)
a = clf.coef_


cm = confusion_matrix(y_test,y_pred)

print(cm)


scores = cross_validation.cross_val_score(clf, X, y, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#%% add the time as featuers

def randomSample(df,df1,label):
    rows = random.sample(df.index,2000)
    df = df.ix[rows]
    timeFeatures = pd.DataFrame()
    timeFeatures['time'] = [5]*2000
    df = df.reset_index()
    df = pd.concat([df,timeFeatures],axis=1)
   # print df.head(4)
   # print timeFeatures.head(4)
    
    rows1 = random.sample(df1.index,2000)
    df3 = df1.ix[rows1]
    timeFeatures1 = pd.DataFrame()

    timeFeatures1['time'] = [1]*2000
    df3 = df3.reset_index()

    df3 = pd.concat([df3,timeFeatures1],axis=1)
    df2 = pd.concat([df,df3],axis=0)
    df2 = df2.reset_index()
    del df2['index']
    labels = pd.DataFrame()
    labels['label'] = [label]*4000
    df2 = pd.concat([df2,labels],axis=1)

    return df2

healthTrain = randomSample(healthData,healthData1,1)
particialTrain = randomSample(particialData,particialData1,2)
hyperTrain = randomSample(hype,hype1,3)

#%%

Data = pd.concat([healthTrain,hyperTrain,particialTrain],axis=0)
Data = Data.reset_index()
#print Data.tail(4)

df = Data.iloc[np.random.permutation(len(Data))]
del df['index']

#%%
col = Data.columns
X = df.loc[:,col[1:-1]]
y = df.loc[:,col[-1]]

del X['level_0']
#%% go the the modeling part
from sklearn.lda import LDA
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
clf = LDA()

X_train, X_test, y_train, y_test = cross_validation.train_test_split( X,y, test_size=0.5, random_state=0)

y_pred = clf.fit(X_train, y_train).predict(X_test)
clf = clf.fit(X_train,y_train)
b= clf.coef_

cm = confusion_matrix(y_test,y_pred)

print(cm)


scores = cross_validation.cross_val_score(clf, X, y, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


#import pylab as plt
#plt.figure()
#plt.plot(clf.coef_[1,:])




























