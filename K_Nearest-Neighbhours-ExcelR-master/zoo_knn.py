# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 19:57:51 2020

@author: HP
"""

import pandas as pd
import numpy as np


zoo = pd.read_csv("D:\\Data Analytics Assignments\\KNN\\Zoo.csv")

from sklearn.model_selection import train_test_split
train,test = train_test_split(zoo,test_size = 0.2, random_state=10) # 0.2 => 20 percent of entire data 

from sklearn.neighbors import KNeighborsClassifier as KNC

k=0
for k in range(0,32):
    if(k%2!=0):
        neigh = KNC(n_neighbors=k)
        neigh.fit(train.iloc[:, 1:17], train.iloc[:,17])
        print('train_accur '+str(k) +' : '+str(np.mean(neigh.predict(train.iloc[:,1:17])==train.iloc[:,17])))   
        print('test_accur '+str(k)+' : '+str(np.mean(neigh.predict(test.iloc[:,1:17])==test.iloc[:,17])))
        print('_'*30)
        
# For K=5 prediction of trained data and test data are close...
        # SOwe consider 5 k nearest neighbors for the classifier