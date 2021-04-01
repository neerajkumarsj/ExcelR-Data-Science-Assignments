# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 03:04:42 2020

@author: HP
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
 
salary_train=pd.read_csv('D:\\Data Analytics Assignments\\Support Vector Machine\\SalaryData_Train(1).csv')
salary_test=pd.read_csv('D:\\Data Analytics Assignments\\Support Vector Machine\\SalaryData_Test(1).csv')

enc=LabelEncoder()
salary_train['workclass']=enc.fit_transform(salary_train.iloc[:,1])
salary_train['maritalstatus']=enc.fit_transform(salary_train.iloc[:,4])
salary_train['occupation']=enc.fit_transform(salary_train.iloc[:,5])
salary_train['relationship']=enc.fit_transform(salary_train.iloc[:,6])
salary_train['race']=enc.fit_transform(salary_train.iloc[:,7])
salary_train['sex']=enc.fit_transform(salary_train.iloc[:,8])
salary_train['native']=enc.fit_transform(salary_train.iloc[:,12])
salary_train.drop(columns='education', axis=1, inplace=True)

salary_test['workclass']=enc.fit_transform(salary_test.iloc[:,1])
salary_test['maritalstatus']=enc.fit_transform(salary_test.iloc[:,4])
salary_test['occupation']=enc.fit_transform(salary_test.iloc[:,5])
salary_test['relationship']=enc.fit_transform(salary_test.iloc[:,6])
salary_test['race']=enc.fit_transform(salary_test.iloc[:,7])
salary_test['sex']=enc.fit_transform(salary_test.iloc[:,8])
salary_test['native']=enc.fit_transform(salary_test.iloc[:,12])
salary_test.drop(columns='education', axis=1, inplace=True)




X_train=salary_train.iloc[:, 0:12]
Y_train=salary_train.iloc[:, 12]


X_test=salary_test.iloc[:, 0:12]
Y_test=salary_test.iloc[:, 12]

#from sklearn.model_selection import train_test_split 
#X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.2, random_state=6)


from sklearn.svm import SVC

model_linear=SVC(kernel='linear')
model_linear.fit(X_train, Y_train)
Y_predict=model_linear.predict(X_test)
np.mean(Y_test==Y_predict)
pd.crosstab(Y_test, Y_predict)


model_rbf=SVC(kernel='rbf')
model_rbf.fit(X_train, Y_train)
Y_predict2=model_rbf.predict(X_test)
np.mean(Y_test==Y_predict2)
pd.crosstab(Y_test, Y_predict2)
