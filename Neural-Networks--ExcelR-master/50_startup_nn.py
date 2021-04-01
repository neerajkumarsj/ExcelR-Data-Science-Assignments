# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 20:21:34 2020

@author: HP
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense

startup_nn=pd.read_csv('D:\\Data Analytics Assignments\\Neural Network\\50_Startups.csv')

state_dummies=pd.DataFrame(pd.get_dummies(startup_nn['State']))

startup_nn=pd.concat([startup_nn, state_dummies], axis=1)
startup_nn.drop(columns='State', axis=1, inplace=True)

startup_model=Sequential()
startup_model.add(Dense(30,input_dim=6,activation="relu"))
startup_model.add(Dense(20,activation="relu"))
startup_model.add(Dense(10,activation="relu"))
startup_model.add(Dense(1, kernel_initializer="normal"))
startup_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

startup_nn=startup_nn.iloc[:, [0,1,2,4,5,6,3]]

column_names=list(startup_nn.columns)
predictors=column_names[:6]
target=column_names[6]

first_model=startup_model
first_model.fit(np.array(startup_nn[predictors]),np.array(startup_nn[target]),epochs=10)
pred_train = startup_model.predict(np.array(startup_nn[predictors]))
pred_train = pd.Series([i[0] for i in pred_train])
rmse_value = np.sqrt(np.mean((pred_train-startup_nn[target])**2))

import matplotlib.pyplot as plt
plt.plot(pred_train,startup_nn[target],"bo")

corref=np.corrcoef(pred_train,startup_nn[target]) #strong correlation


from ann_visualizer.visualize import ann_viz
ann_viz(first_model, title='My Third Neural Network', view=True)

