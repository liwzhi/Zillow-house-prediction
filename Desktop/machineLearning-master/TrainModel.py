# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 15:25:58 2015

@author: weizhi
"""


from datetime import datetime
from csv import DictReader
from math import exp, log, sqrt
from random import random
import pickle

# B, model
alpha = .005  # learning rate
beta = 1.   # smoothing parameter for adaptive learning rate
L1 = 0.     # L1 regularization, larger value means more regularized
L2 = 1.     # L2 regularization, larger value means more regularized

# C, feature/hash trick
D = 1881            # number of weights to use
interaction = False     # whether to enable poly2 feature interactions

# D, training/validation
epoch = 1       # learn training data for N passes
holdafter = 9   # data after date N (exclusive) are used as validation
holdout = None  # use every N training instance for holdout validation

#%% Train a model 
class gradient_descent(object):

    def __init__(self, alpha, beta, L1, L2, D, interaction):
        # parameters
        self.alpha = alpha
        self.beta = beta
        self.L1 = L1
        self.L2 = L2

        # feature related parameters
        self.D = D
        self.interaction = interaction

        # model
        # G: squared sum of past gradients
        # w: weights
        self.w = [0.] * D  
        self.G = [0.] * D 


    def predict(self, x):
        ''' Get probability estimation on x

            INPUT:
                x: features

            OUTPUT:
                probability of p(y = 1 | x; w)
        '''
        w = self.w	
        # wTx is the inner product of w and x
        wTx = 0.
        for i in range(x.shape[0]):
            item = x[i]
            wTx += w[i]*item
      #  print wTx

        # bounded sigmoid function, this is the probability estimation
        return 1. / (1. + exp(-max(min(wTx, 35.), -35.)))

    def update(self, x, p, y):
        ''' Update model using x, p, y

            INPUT:
                x: feature, a list of indices
                p: click probability prediction of our model
                y: answer

            MODIFIES:
                self.G: increase by squared gradient
                self.w: weights
        '''
        # parameters
        alpha = self.alpha
        beta = self.beta
        L1 = self.L1
        L2 = self.L2

        # model
        w = self.w
        G = self.G

        # gradient under logloss
        g = p - y
        # update z and n
        for i in range(x.shape[0]):
            G[i] += g*g
#            w[i] -= alpha*1/sqrt(n[i]) * (g - L2*w[i]) ## Learning rate reducing as 1/sqrt(n_i) : ALso gives good performance but below code gives better results
            w[i] -= alpha/(beta+sqrt(G[i])) * (g - L2*w[i]) ## Learning rate reducing as alpha/(beta + sqrt of sum of g_i)

        self.w = w
        self.G = G


#%%
def logloss(p, y):
    ''' FUNCTION: Bounded logloss

        INPUT:
            p: our prediction
            y: real answer

        OUTPUT:
            logarithmic loss of p given y
    '''

    p = max(min(p, 1. - 10e-15), 10e-15)
    return -log(p) if y == 1. else -log(1. - p)
 
#%%
def data(path, D):
    ''' GENERATOR: Apply hash-trick to the original csv row
                   and for simplicity, we one-hot-encode everything

        INPUT:
            path: path to training or testing file
            D: the max index that we can hash to

        YIELDS:
            ID: id of the instance, mainly useless
            x: a list of hashed and one-hot-encoded 'indices'
               we only need the index since all values are either 0 or 1
            y: y = 1 if we have a click, else we have y = 0
    '''

    for t, row in enumerate(DictReader(open(path), delimiter=',')):
        # process id
        #print row
        
        try:
            ID=row['ID']
            del row['ID']
        except:
            pass
        # process clicks
        y = 0.
        target='target'#'IsClick' 
        if target in row:
            if row[target] == '1':
                y = 1.
            del row[target]

        # extract date

        # turn hour really into hour, it was originally YYMMDDHH

        # build x
        x = []
        for key in row:
            value = row[key]

            # one-hot encode everything with hash trick
            index = abs(hash(key + '_' + value)) % D
            x.append(index)

        yield ID,  x, y



           
#%% read the data             
learner = gradient_descent(alpha, beta, L1, L2, D, interaction)
count =0
# start training
for e in range(epoch):
    loss = 0.
    count = 0
    for index in range(X_scaled.shape[0]):
        X = X_scaled[index,:]
        y_train = y.iloc[index]
        p = learner.predict(X)
        loss +=logloss(p,y_train)
        count+=1

        if count%1000==0:
            #print count,loss/count
            print('%s\tencountered: %d\tcurrent logloss: %f' % (
                datetime.now(), count, loss/count))
       # if count>10000: # comment this out when you run it locally.
       #     break      