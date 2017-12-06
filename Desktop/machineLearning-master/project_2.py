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
       # print nameFile
        filePaths.append(pathFolder  + '/'+nameFile[0])
    return (filePaths,keys)

rdfPath = '/Users/weizhi/Downloads/cache/epub'
dataFile,keys = findFilePath(rdfPath)


#%%raw_data = '/Users/weizhi/Downloads/cache/epub/2801/pg2801.rdf' 

import rdflib
import datetime
import urllib2

class dataPath(object):
   # def __ini__(self,publishTime=[],textContent=[],subject=[],predicate=[],obj=[]):
#        self.time = publishTime
#        self.textContent = textContent 
#        self.subject = subject
#        self.predicate = predicate
#        self.obj = obj
    
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
        endString =  key + '.txt'
        URL = []
        for item in subject:
            if item.endswith(endString):
                URL.append(item)
        if len(URL) !=0:
            try:
                data = urllib2.urlopen(URL[0]).read()# read only 20 000 chars
                data = data.split("\n") # then split it into lines
                return data
            except:
                pass
           # return data
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
for index in range(10):
    result = obj.graphParse(dataFile[index])
    timePub = obj.publishTimeGet(result[0])
    rawText = obj.readURL(result[1],keys[index])
    Dict = obj.metaData(rawText)
   # Dict['key'] = keys[index]
    print Dict 
    DictOutput.append(Dict)
#print result[1]
#%%print timePub
import nltk
textBooks = nltk.corpus.gutenberg.fileids()



#import urllib2

#data = urllib2.urlopen(URL[0]).read()# read only 20 000 chars
#data = data.split("\n") # then split it into lines

#result = obj.graphParse(dataFile[0])        

#%%

    
#%%
#filename = 'cache/epub/78/pg78.rdf'
#from lxml import etree
#rdf = open(raw_data).read()
#tree = etree.fromstring(rdf)
#resource_tag = '{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource'
#hasFormat_tag = './/{http://purl.org/dc/terms/}hasFormat'
#resources = [el.attrib[resource_tag] for el in tree.findall(hasFormat_tag)]
#urls = [url for url in resources if url.endswith('htm')]
##// urls[0] is 'http://www.gutenberg.org/files/78/78-h/78-h.htm'
#




#%%

#http://stackoverflow.com/questions/19588952/python-reading-rdf-files-scraping-gutenberg-books


#print ', '.join(output)

#from lxml import etree
#rdf = open(raw_data).read()
#tree = etree.fromstring(rdf)
#resource_tag = '{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource'
#hasFormat_tag = './/{http://purl.org/dc/terms/}hasFormat'
#resources = [el.attrib[resource_tag] for el in tree.findall(hasFormat_tag)]
#urls = [url for url in resources if url.endswith('htm')]
## urls[0] is 'http://www.gutenberg.org/files/78/78-h/78-h.htm'