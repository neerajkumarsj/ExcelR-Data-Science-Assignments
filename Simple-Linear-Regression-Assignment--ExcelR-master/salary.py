# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 11:30:26 2020

@author: HP
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import statsmodels.formula.api as smf

def mse(y_pred, y_actual):
    a=y_pred
    b=y_actual
    mse=np.square(np.subtract(a, b)).mean()
    return mse



salary=pd.read_csv("E:/Neeraj/bro.csv")
salary

salary.corr()

scale = MinMaxScaler()
scaled = scale.fit_transform(salary)
inv_scaled = scale.inverse_transform(scaled)

plt.hist(salary.rank)
plt.boxplot(salary.rank)
plt.hist(salary.score)   #As there are less numbers of data histogram is not showing good result, no bell curve, less data density

model=smf.ols("rank~score",data=salary).fit()
model
model.summary()
pred = model.predict(salary)
pred = np.array(pred)
y_rank = np.array(salary['rank']).astype(float)
mse_1=mse(pred, y_rank)
mse_1       #least mean square error

plt.scatter(salary.score, salary.rank, color='red'); plt.plot(salary.rank, pred, color='black');plt.xlabel('experience in year');plt.ylabel('salary')
#According to observations this is the best model for the data.......**solved
'''**solved'''

model2=smf.ols('rank~np.log(exp)', data=salary).fit()
model2
model2.summary()
pred2=model2.predict(rankary)
pred2
plt.scatter(rankary.exp, salary.rank, color='red');plt.plot(rankary.exp, pred2, color='black')
mse_2=mse(pred2, salary.rank)
mse_2


model3=smf.ols('np.log(rank)~exp', data=salary).fit()
model3
model3.summary()

pred3_log=model3.predict(rankary)
pred3=np.exp(pred3_log)

plt.scatter(rankary.exp, salary.rank, color='red');plt.plot(rankary.exp, pred3, color='black')

mse_3=mse(pred3, salary.rank)
mse_3

#If we consider AIC value we can take this model too as the prediction model