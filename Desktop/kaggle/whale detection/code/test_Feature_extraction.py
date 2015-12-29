# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 20:47:44 2015

@author: weizhi
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 16:08:38 2015

@author: weizhi
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 21:16:49 2015

@author: weizhi
"""


import pandas as pd

testLabel = pd.read_csv('/Users/weizhi/Desktop/kaggle/whale detection/sample_submission.csv')

import site; site.addsitedir("/usr/local/lib/python2.7/site-packages")
import cv2




#%%
import pylab as plt


#%%
import numpy as np



#%%
from scipy import ndimage as nd

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

# https://docs.python.org/2/library/os.html
def findFilePath(path):
    os.chdir(path)
    filePaths = []
    for file in glob.glob("*.jpg"):
        filePaths.append(file)
    return filePaths

#%% hours and generate the outputs
# deal with each csv file 
row = 1000
col = 1000
surf = cv2.SURF()
orb = cv2.ORB()

fea_det = cv2.FeatureDetector_create("SIFT")
des_ext = cv2.DescriptorExtractor_create("SIFT")
def readCSV(path, filePaths,row,col,trainLabel):
    des_list = []
    fea_det = cv2.FeatureDetector_create("SIFT")
    des_ext = cv2.DescriptorExtractor_create("SIFT")
   # number = 500    
  #  data = np.zeros([1000,1000,len(filePaths)])
   # kpList = []
  #  desList = []    
    for i in range(len(filePaths)):
        print i
        filePath = path + '/' + filePaths[i]
        try:
            img1 =cv2.imread(filePath,0)
            img1 = cv2.resize(img1,(500,500))
            img1 = img1.astype(np.uint8)
            kpts = fea_det.detect(img1)
            kpts, des = des_ext.compute(img1,kpts)
            des_list.append((trainLabel['Image'][i],des))
            print trainLabel['Image'][i]

        except:
            print "wrong image"
         #   img1 = np.zeros((100,100))

      #  data[:,:,i] = img1
    return des_list
 #,kpList,desList




#%% testing
path = '/Users/weizhi/Desktop/kaggle/whale detection/imgs'
filePaths = findFilePath(path)


#%%
#%% get the features
des_list = readCSV(path, testLabel['Image'],row,col,testLabel)


#%% use the k-means
descriptors = des_list[0][1]
for img_paths, descriptor in des_list[1:]:
    try:
        descriptors = np.vstack((descriptors, descriptor))  
    except:
        print "wrong image"

from scipy.cluster.vq import *    
k = 300
voc, variance = kmeans(descriptors,k,1)

im_features = np.zeros((len(testLabel['Image']),k),'float32')
for i in range(len(trainLabel['Image'])):
    try:
        words, distance = vq(des_list[i][1],voc)
        for w in words:
            im_features[i][w] +=1
    except:
        print "error message"
#%% perform the Tf-Idf vectorization 

nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
#idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')
#%% scaling the words
from sklearn.preprocessing import StandardScaler

stdSlr = StandardScaler().fit(im_features)
im_features = stdSlr.transform(im_features)
#%%
data = pd.DataFrame(im_features)
raw_feature = pd.DataFrame(descriptors)
raw_feature.to_csv('/Users/weizhi/Desktop/kaggle/whale detection/RAWtest_features.csv',index=False)

data.to_csv('/Users/weizhi/Desktop/kaggle/whale detection/test_features.csv',index=False)
