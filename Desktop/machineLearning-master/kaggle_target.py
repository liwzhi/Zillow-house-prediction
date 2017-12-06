# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 17:05:28 2015

@author: weizhi
"""

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
#testData = pd.read_csv('/Users/weizhi/Downloads/kaggle competion/test.csv')
trainData = pd.read_csv('/Users/weizhi/Downloads/kaggle competion/train.csv')

v = DictVectorizer(sparse=False)

X = v.fit_transform(trainData.iloc[:,1])
