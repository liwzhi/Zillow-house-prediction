# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 17:47:27 2015

@author: weizhi
"""

#%% loading each files
import glob, os
import collections
# https://docs.python.org/2/library/os.html
def findFilePath(path):
    os.chdir(path)
    filePaths = []
    for file in glob.glob("*.csv"):
        filePaths.append(file)
    return filePaths

#%% hours and generate the outputs
# deal with each csv file 
import pandas as pd

def readCSV(filePath):
    data = pd.read_csv(filePath)
    return data

def transfromTimeFormat(data,n):
    timeColumn = data[data.keys()[-1]]
    for i in range(len(timeColumn)):
        item = timeColumn.loc[i]
        T1 = item.split(' ')
        T2 = T1[1].split(':')[0]
        if int(T2)%2==0:
            T3 = T2 +  '_' + str(int(T2)+n).zfill(2)
        else:
            T3 = T2 + '_' + str(int(T2)-n).zfill(2)
        T4 = T1[0] + '_'+ T3
        timeColumn.loc[i] = T4
    data[data.keys()][-1] = timeColumn
    return data
    
def generateOutputs(data,path,fileName):
    keys = [key for key in data.keys()]
    keys.reverse()
    dataGroup = data.groupby(keys).groups
    keysOuput = sorted(dataGroup.iterkeys())  # keep the keys sorted rather than hashing
    outputs = pd.DataFrame(columns = ['period','content_id','uid','count'])
    count = 0
    for key in keysOuput:
        curr = list(key)   # write to each columns to outputs.csv
        curr.append(len(dataGroup[key])) # get the count of keys from groupby
        outputs.loc[count] = curr
        count +=1 
    savePath = path + '/' + 'Outputs'+'/'+'Output_'+fileName
    print savePath
    outputs.to_csv(savePath,index=False)
    return outputs

#%% testing
path = '/Users/weizhi/Desktop/everStringProject'
filePaths = findFilePath(path)
data = readCSV(path + '/'+ filePaths[0])
dataTransform = transfromTimeFormat(data,1)
outputs = generateOutputs(dataTransform,path,filePaths[0])

