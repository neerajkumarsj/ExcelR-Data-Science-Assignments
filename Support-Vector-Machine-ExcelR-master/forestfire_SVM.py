# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 21:13:05 2020

@author: HP
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
forestfire=pd.read_csv('D:\\Data Analytics Assignments\\Support Vector Machine\\forestfires.csv')

X=forestfire.iloc[:, 2:30]
Y=forestfire.iloc[:, 30]

from sklearn.model_selection import train_test_split 
X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.2, random_state=6)


from sklearn.svm import SVC

model_linear=SVC(kernel='linear')
model_linear.fit(X_train, Y_train)
Y_predict=model_linear.predict(X_test)
np.mean(Y_test==Y_predict)
pd.crosstab(Y_test, Y_predict)


model_rbf=SVC(kernel='rbf', gamma='scale')
#help(SVC)
model_rbf.fit(X_train, Y_train)
Y_predict2=model_rbf.predict(X_test)
np.mean(Y_test==Y_predict2)
pd.crosstab(Y_test, Y_predict2)
