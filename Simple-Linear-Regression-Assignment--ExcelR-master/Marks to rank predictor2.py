# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 14:35:54 2020

@author: Admin
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error
import sklearn
import matplotlib.pyplot as plt

rank = pd.read_csv('E:/Neeraj/bro.csv')
rank.corr()



y = np.array(rank['rank'])
X = np.array(rank['score'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# Create new axis for x column
X_train = X_train[:,np.newaxis]
X_test = X_test[:,np.newaxis]
y_t = y_train[:,np.newaxis]



scale = StandardScaler()
scaled = scale.fit_transform(X_train)
scaled_test = scale.fit_transform(X_test)
scaled_y = scale.fit_transform(y_t)


#x_dummy = scale.inverse_transform(scaled)
#y_dummy = scale.inverse_transform(scaled_y)

model = LinearRegression().fit(scaled, scaled_y)
pred = model.predict(scaled)
pred = abs(scale.inverse_transform(pred))
y_pred = model.predict(scaled_test)
y_pred = abs(scale.inverse_transform(y_pred))

score = r2_score(pred, y_train)
score2 = r2_score(y_pred, y_test)
# calculate Mean square error
mse = mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mse)

Ypred = pd.DataFrame()
Ypred["actual data"] = y_test
Ypred["predicted data"] = y_pred

X_test.shape
y_pred.shape
X_test = np.squeeze(X_test)
test = pd.DataFrame()
test["given data"] = X_test
test["predicted data"] = y_pred

X_train = np.squeeze(X_train)
train = pd.DataFrame()
train["given data"] = X_train
train["predicted data"] = pred
# Plotting the actual and predicted values

from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
lasso=Lasso()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
lasso_regressor=GridSearchCV(lasso,parameters,scoring='neg_mean_squared_error',cv=5)

lasso_regressor.fit(scaled,scaled_y)
print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)
prediction_lasso_train = lasso_regressor.predict(scaled)
prediction_lasso_train = abs(scale.inverse_transform(prediction_lasso_train))
prediction_lasso=lasso_regressor.predict(scaled_test)
prediction_lasso = abs(scale.inverse_transform(prediction_lasso))

score_l = r2_score(prediction_lasso_train, y_train)
score_l2 = r2_score(y_pred, y_test)