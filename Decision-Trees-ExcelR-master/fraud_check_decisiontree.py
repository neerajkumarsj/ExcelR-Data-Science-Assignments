# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 16:13:33 2020

@author: HP
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import scale

fraud=pd.read_csv('D:\\Data Analytics Assignments\\Decision Tree Classifier\\Fraud_check.csv')

#categorizing taxable income 
bins=[10000,30000,100000]
group_names=['Risky', 'Good']    
fraud['Tax_status']=pd.cut(fraud['Taxable.Income'], bins, labels=group_names)

#categorical data labeling
enc=LabelEncoder()
fraud['Undergrad']=enc.fit_transform(fraud.iloc[:, 0])
fraud['Marital.Status']=enc.fit_transform(fraud.iloc[:, 1])
fraud['Urban']=enc.fit_transform(fraud.iloc[:, 5])

predictors=fraud.iloc[:,[0,1,3,4,5]]
target=fraud.iloc[:,6]

#scaling data
predictors_scale=scale(predictors)


#spliting data
from sklearn.model_selection import train_test_split
predictors_train, predictors_test, target_train, target_test=train_test_split(predictors_scale, target, test_size=0.2, random_state=6)


from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier(criterion="entropy")
model.fit(predictors_train, target_train)

target_predict=model.predict(predictors_test)
np.mean(target_predict==target_test)
