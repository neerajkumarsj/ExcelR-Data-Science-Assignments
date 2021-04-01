# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 21:03:51 2020

@author: HP
"""

import pandas as pd
import numpy as np

salary_train=pd.read_csv('D:\\Data Analytics Assignments\\Naive Bayes\\SalaryData_Train.csv')
salary_test=pd.read_csv('D:\\Data Analytics Assignments\\Naive Bayes\\SalaryData_Test.csv')

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

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

model_gnb=GaussianNB()
model_gnb.fit(X_train, Y_train)
Y_pred=model_gnb.predict(X_test)
np.mean(Y_pred==Y_test)
conf_gnb=confusion_matrix(Y_pred, Y_test)
conf_gnb


model_mnb=MultinomialNB()
model_mnb.fit(X_train, Y_train)
Y_pred2=model_mnb.predict(X_test)
np.mean(Y_pred2==Y_test)
conf_mnb=confusion_matrix(Y_pred2, Y_test)
conf_mnb





















