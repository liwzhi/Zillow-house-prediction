# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 14:18:31 2015

@author: weizhi
"""
import glob, os
import re
# https://docs.python.org/2/library/os.html
def findFilePath(dataPath):
    """
    Input: dataPath: RDF folder path
          From: http://www.gutenberg.org/wiki/Gutenberg%3aFeeds#Current_RDF_Format     
    Output:
            Each rdf file path
    
    """
    dataFileName = os.listdir(dataPath)
    filePaths = []
    keys = []
    for i in range(1,len(dataFileName)):
        item = dataFileName[i]
        keys.append(item)
        pathFolder = dataPath + '/' + item
      #  print pathFolder
        nameFile = os.listdir(pathFolder)
        for item in nameFile:
            if item.endswith('.rdf'):
                filePaths.append(pathFolder  + '/'+item)
    return (filePaths,keys)

rdfPath = '/Users/weizhi/Downloads/cache/epub'
dataFile,keys = findFilePath(rdfPath)


#%%raw_data = '/Users/weizhi/Downloads/cache/epub/2801/pg2801.rdf' 

import rdflib
import datetime
import urllib2
import pandas as pd
import pylab as pl
class dataPath(object):    
    def graphParse(self,rdfPath):
        graph = rdflib.Graph()
        graph.parse(rdfPath)
        obj = []
        subject = []
        predicate= []
        for s, p, o in graph:
           # print s,p,o # subject, predicate, object            
            if type(o) == rdflib.term.Literal:
                obj.append(o.toPython())
            
            if type(o) == rdflib.term.URIRef:
                subject.append(o.toPython())
            if type(o) == rdflib.term.BNode:
                predicate.append(o.toPython())    
        #time = self.publishTimeGet(obj)
       # content = self.readURL(subject)        
        return obj,subject,predicate
    def publishTimeGet(self,obj):
    
    
        time = []
        for item in obj:
            if type(item) == datetime.date:
                time.append(item)
        return time
    def readURL(self,subject,key):
        """
        https://gist.github.com/andreasvc/b3b4189120d84dec8857
        """
        endString =  '.txt'
        URL = []
        data = []
        for item in subject:
            if item.endswith(endString):
                URL.append(item)
        if len(URL) !=0:
            try:
                data = urllib2.urlopen(URL[0]).read()# read only 20 000 chars
                data = data.split("\n") # then split it into lines
               # return data
            except:
                pass
        if len(data) !=0:
            return data
        else:
            return URL
    def metaData(self,data):
      #  for line in data
        Dict = {}
        tag = ['author','title','release date']
        count = 0
        index = 0
        for line in data:
            for item in tag:
                if line.lower().startswith(item+':'):
                    Dict[item] = line.split(':')[1].rstrip()
                    count +=1
            index +=1
            if count ==3 or index>200:
                return Dict
                break
            
        return Dict


            
        
obj = dataPath()
DictOutput = []
Dict = {}

for index in range(100):
    result = obj.graphParse(dataFile[index])
    timePub = obj.publishTimeGet(result[0])
    rawText = obj.readURL(result[1],keys[index])
   # Dict = obj.metaData(rawText)
   # Dict['key'] = keys[index]
 #   print Dict 0000
#    DictOutput[keys[index]] = timePub
    DictOutput.append(timePub)
    

#%%
day = {}
month = {}
year = {}

for item in DictOutput:
    try:
        currDay = item[0].isoformat()
        currMonth = item[0].isoformat()[:-3]
        currYear = item[0].isoformat()[:-6]
        if currDay in day.keys():
            day[currDay] +=1
        else:
            day[currDay] = 1
        if currMonth in month.keys():
            month[currMonth]+=1
        else:
            month[currMonth] =1
        if currYear in year.keys():
            year[currYear] +=1
        else:
            year[currYear] =1
    except:
        pass

#%% Plot the figures


def histPlot(day,name):
    day = pd.DataFrame(day.items(),columns = ['Date','Count']).sort(['Date'])
    #print day
    day.plot(x='Date',y='Count',legend=False)
    pl.title(name + ' -- publish books')
histPlot(day,'Day')   
histPlot(year,'Year')
histPlot(month,'Month')


#%% using the 
import nltk
from numpy import random
from gutenberg.acquire import load_etext
from gutenberg.cleanup import strip_headers


bookNumber = set(random.randint(10,50024,size=2000))
#f.write(foo.encode('utf8'))

metaInfo = []

for item in bookNumber:
   # print item
    try: 
       # print item
        # loading the raw txt 
        data = load_etext(item).split("\n")
        
        # save the txt data path
        filePath = rdfPath + '/' +str(item) + '/' +  str(item) + '.txt'
        f = open(filePath,'w')
        f.write(data.encode('utf8'))
        f.close()
        # get the meta data 
        Dict = obj.metaData(data)
        metaInfo.append((Dict,filePath))
        print len(metaInfo)
    except:
        continue
#%%do the data mining to these txt, author, title, release time, etc, need time to work on this part
        
        
        
        
        
        
        
        
        
        