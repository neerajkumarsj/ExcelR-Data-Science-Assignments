# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 19:58:22 2020

@author: HP
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot
from statsmodels.graphics.gofplots import qqplot

glass = pd.read_csv("D:\\Data Analytics Assignments\\KNN\\glass.csv")

glass['Mg'].hist()
glass['RI'].hist()
glass['Na'].hist()
glass['Al'].hist()
glass['K'].hist()
glass['Ca'].hist()
glass['Ba'].hist()
qqplot(glass['Mg'])
np.log(glass['Mg'])
np.exp(glass['Mg'])
qqplot(np.sqrt(glass['Mg']))


from sklearn.model_selection import train_test_split
train,test = train_test_split(glass,test_size = 0.2,) # 0.2 => 20 percent of entire data 

from sklearn.neighbors import KNeighborsClassifier as KNC

k=0
for k in range(0,32):
    if(k%2!=0):
        neigh = KNC(n_neighbors=k)
        neigh.fit(train.iloc[:, 0:9], train.iloc[:,9])
        print('train_accur '+str(k) +' : '+str(np.mean(neigh.predict(train.iloc[:,0:9])==train.iloc[:,9])))   
        print('test_accur '+str(k)+' : '+str(np.mean(neigh.predict(test.iloc[:,0:9])==test.iloc[:,9])))
        print('_'*30)

#For K=29 test data accutacy and train data accuracy are close