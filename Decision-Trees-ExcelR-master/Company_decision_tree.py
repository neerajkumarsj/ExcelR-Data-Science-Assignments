# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 22:46:36 2020

@author: HP
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import scale


company=pd.read_csv('D:\\Data Analytics Assignments\\Decision Tree Classifier\\Company_Data.csv')
Enc=LabelEncoder()
company_enc=company
company_enc['ShelveLoc']=Enc.fit_transform(company.iloc[:, 6])
company_enc['Urban']=Enc.fit_transform(company.iloc[:, 9])
company_enc['US']=Enc.fit_transform(company.iloc[:, 10])

bins=[-1,6,12,18]
group_names=['low_sales','moderate_sales','high_sales']
company_enc['Sale_status']=pd.cut(company_enc['Sales'], bins, labels=group_names)



predictors=company_enc.iloc[:, [1,3,4,5]]
target=pd.DataFrame(company_enc.iloc[:, 11])

predictors_scales=pd.DataFrame(scale(predictors))


'''from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300, random_state=6)
y_kmeans=kmeans.fit(target)
kmeans.labels_
target=pd.Series(kmeans.labels_)'''


from sklearn.model_selection import train_test_split

predictors_train, predictors_test, target_train, target_test=train_test_split(predictors_scales, target, test_size=0.2, random_state=6)
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier(criterion="entropy")
model.fit(predictors_train, target_train)

target_predict=model.predict(predictors_test)
np.mean(target_predict==target_test.Sale_status)
pd.crosstab(target_test.Sale_status,target_predict)
pd.Series(target_predict).value_counts()

help(pd.DataFrame)
