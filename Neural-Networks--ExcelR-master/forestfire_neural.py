# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 00:03:48 2020

@author: HP
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense

forestfire_nn=pd.read_csv('D:\\Data Analytics Assignments\\Neural Network\\forestfires.csv')
forestfire_nn.drop(columns=['month','day'], axis=1, inplace=True)

forest_model=Sequential()
forest_model.add(Dense(90,input_dim=28,activation="relu"))
forest_model.add(Dense(50,activation="relu"))
forest_model.add(Dense(20,activation="relu"))
forest_model.add(Dense(1, kernel_initializer="normal"))
forest_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])


enc=LabelEncoder()
forestfire_nn['size_category']=enc.fit_transform(forestfire_nn.size_category)

column_names=list(forestfire_nn.columns)
predictors=column_names[:28]
target=column_names[28]


first_model=forest_model
first_model.fit(np.array(forestfire_nn[predictors]),np.array(forestfire_nn[target]),epochs=10)
pred_train = first_model.predict(np.array(forestfire_nn[predictors]))
pred_train = pd.Series([i[0] for i in pred_train])
rmse_value = np.sqrt(np.mean((pred_train-forestfire_nn[target])**2))

np.mean(pred_train)  



for i in range(0,517):
    if pred_train[i]<=0.7:
        pred_train[i]=0
    else:
        pred_train[i]=1
        
from sklearn.metrics import confusion_matrix
conf=confusion_matrix(pred_train, forestfire_nn[target])
conf
import matplotlib.pyplot as plt

plt.plot(pred_train, forestfire_nn[target], 'bo')
np.corrcoef(pred_train, forestfire_nn[target])

from ann_visualizer.visualize import ann_viz
ann_viz(first_model, title="My second neural network", view=True)

