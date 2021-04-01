# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 22:15:23 2020

@author: HP
"""

import pandas as pd
#import seaborn as sb
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
#from mlxtend.frequent_patterns import apriori,association_rules
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as ttsplt

bank=pd.read_csv('C:\\Users\\HP\\Desktop\\Assignment7 (Logistic Regression)\\bank-full.csv', sep=';')



'''bank2=pd.DataFrame(bank_pd.split(';'))
bank=[]
with open('C:\\Users\\HP\\Desktop\\Assignment7 (Logistic Regression)\\bank-full.csv') as f:
    bank=f.read()

bank_pd[['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan','contact', 'day', 'month', 'duration',\
      'campaign', 'pdays', 'previous', 'poutcome', 'y']]=bank_pd.apply(lambda x: pd.Series(str(x).split(';')))
bank=bank_pd.split(';')'''


bank_dummies=pd.get_dummies(bank[['job', 'marital', 'education', 'default','housing', 'loan','contact','month','poutcome', 'y']])
bank.drop(['job', 'marital', 'education', 'default','housing', 'loan','contact','month','poutcome', 'y'],inplace=True, axis=1)
bank_new=pd.concat([bank, bank_dummies], axis=1)
bank_new

#sb.countplot(x='job', data=bank)
#pd.crosstab(bank.job, bank.education).plot(kind='bar')

bank.isnull().sum()

coef=bank_new.corr()
coef

X=bank_new.iloc[:, :51]
Y=bank.iloc[:, 16]
sc=StandardScaler()
X_sc=sc.fit_transform(X)

L_enc=LabelEncoder()
Y_label=L_enc.fit_transform(Y)

classifier_full=LogisticRegression()
classifier_full.fit(X_sc, Y_label)                          #logistic regression model
Y_pred_full=classifier_full.predict(X_sc)                   #predicting dependent variables with the model

from sklearn.metrics import confusion_matrix
conf_full=confusion_matrix(Y_label, Y_pred_full)            #using confusion matrix to finde outtrue positive or true negative or the accuracy of predicted output w.r.t actual output
conf_full


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

fpr1, tpr1, thresholds1=roc_curve(Y_label, Y_pred_full)
plt.plot(fpr1, tpr1, color='red', label='roc1')
auc1=roc_auc_score(Y_label, Y_pred_full)


#train test split tomake the regression model with training data and test it using test data
X_train, X_test, Y_train, Y_test = ttsplt(X_sc, Y_label, test_size=0.25, random_state=0)

classifier=LogisticRegression()
classifier.fit(X_train,Y_train)
Y_pred=classifier.predict(X_test)

conf=confusion_matrix(Y_test, Y_pred)
conf


fpr, tpr, thresholds=roc_curve(Y_test, Y_pred)
plt.plot(fpr, tpr, color='red', label='roc')
auc=roc_auc_score(Y_test, Y_pred)
