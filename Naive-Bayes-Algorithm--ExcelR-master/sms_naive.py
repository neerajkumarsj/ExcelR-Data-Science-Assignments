# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 21:04:35 2020

@author: HP
"""

import numpy as np
import pandas as pd
#import chardet
import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
#with open(r'D:\\Data Analytics Assignments\\Naive Bayes\\sms_raw_NB.csv','rb') as f:
#    data=chardet.detect(f.read())
#    sms_nb=pd.read_csv(r'D:\\Data Analytics Assignments\\Naive Bayes\\sms_raw_NB.csv', encoding=data['encoding'])     

#with open("D:\\Data Analytics Assignments\\Naive Bayes\\sms_raw_NB.csv","r") as f:
#    sms_nb = f.read()
#sms=open("D:\\Data Analytics Assignments\\Naive Bayes\\sms_raw_NB.csv", errors='ignore') 
 
#with open("D:\\Data Analytics Assignments\\Naive Bayes\\sms_raw_NB.csv", errors='ignore') as f:
#    sms_nb=f.read()
sms_nb=pd.read_excel('D:\\Data Analytics Assignments\\Naive Bayes\\sms_raw_NB.xlsx')



X=sms_nb.text
Y=sms_nb.type
X_train, X_test, Y_train, Y_test=train_test_split(X,Y, test_size=0.2)
sms_word=" ".join(X)

model_mnb=MultinomialNB()
model_gnb=GaussianNB()
model_gnb.fit(X_train, Y_train)
#sms_words=" ".join(sms_nb['text'])

sms_string=re.sub("[^A-Za-z" "]+"," ",sms_nb).lower()
sms_string=re.sub("[0-9" "]+"," ",sms_nb)

sms_word=sms_string.split(",")
sms_word=" ".join(sms_word)
sms_word=sms_word.split("\n")
sms_word=" ".join(sms_word)
sms_word=sms_word.split(" ")


remove_words=['ham', 'spam', 'type', 'text']

sms_words=[i for i in sms_word if i not in remove_words]


