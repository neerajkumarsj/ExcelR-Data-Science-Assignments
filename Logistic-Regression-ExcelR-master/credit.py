# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 20:44:27 2020

@author: HP
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

credit=pd.read_csv('C:\\Users\\HP\\Desktop\\Assignment7 (Logistic Regression)\\creditcard.csv')
credit

credit.drop(['Unnamed: 0'], inplace=True, axis=1) #dropping index column named unnamed: 0

L_enc=LabelEncoder()                               #need to label encode of categorical columns
card_enc=pd.DataFrame(L_enc.fit_transform(credit.card), columns=['card_enc'])
owner_enc=pd.DataFrame(L_enc.fit_transform(credit.owner), columns=['owner_enc'])
selfemp_enc=pd.DataFrame(L_enc.fit_transform(credit.selfemp), columns=['selfemp_enc'])


credit_new=pd.concat([credit,card_enc,owner_enc,selfemp_enc], axis=1)    #concatenated with previous dataframe with label encoded columns
credit_new.drop(['card','owner', 'selfemp'], inplace=True, axis=1)        #dropping categorical columns as well

X=credit_new.iloc[:,[0,1,2,3,4,5,6,7,8,10,11]]
Y=credit_new.iloc[:,9]

classifier=LogisticRegression()
classifier.fit(X,Y)     
Y_pred=classifier.predict(X)

conf=confusion_matrix(Y, Y_pred)
conf

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
fpr, tpr, thresholds=roc_curve(Y, Y_pred)
plt.plot(fpr, tpr, color='red', label='roc')
auc=roc_auc_score(Y, Y_pred)


SC=StandardScaler()                         #After standardizing the independent variables
X_sc=SC.fit_transform(X)
classifier2=LogisticRegression()
classifier2.fit(X_sc,Y)
Y_pred2=classifier2.predict(X_sc)

conf2=confusion_matrix(Y, Y_pred2)
conf2

fpr2, tpr2, thresholds2=roc_curve(Y, Y_pred2)
plt.plot(fpr2, tpr2, color='red', label='roc2')
auc2=roc_auc_score(Y, Y_pred2)