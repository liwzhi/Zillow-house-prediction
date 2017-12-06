# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 21:46:53 2015

@author: weizhi
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
Dict['1B'] = ['/Users/weizhi/Downloads/TrainingData/1B/3-11-13_09.59.57/']
Dict['1B'].append('/Users/weizhi/Downloads/TrainingData/1B/3-11-13_09.59.28/')

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


healthData = readCSV(healthPath[0])
particialData = readCSV(particial[0])
hype = readCSV(hype[0])


#%%%%%%%
Dict['1B1hr'] = ['/Users/weizhi/Downloads/TrainingData/1B1hr/2013-11-13_10.48.24/']
Dict['1B1hr'].append('/Users/weizhi/Downloads/TrainingData/1B1hr/2013-11-13_10.48.50/')

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


healthData1 = readCSV(healthPath1[0])
particialData1 = readCSV(particial1[0])
hype1 = readCSV(hype1[0])

#%% concate the data for the training sets
def randomSample(df):
    rows = random.sample(df.index,1000)
    return df.ix[rows]


healthData_2000 = randomSample(healthData)
particial_2000 = randomSample(particialData)
hyper_2000 = randomSample(hype)


healthData1_2000 = randomSample(healthData1)
particial1_2000 = randomSample(particialData1)
hyper1_2000 = randomSample(hype1)


#%%  build the features 
healthTrain = pd.concat([healthData_2000.reset_index(),healthData1_2000.reset_index()],axis=0)

del healthTrain['index']

labels = pd.DataFrame()
labels = [1]*2000

