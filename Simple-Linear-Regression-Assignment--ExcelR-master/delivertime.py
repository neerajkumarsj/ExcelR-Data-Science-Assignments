# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 00:49:03 2020

@author: HP
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn import preprocessing

def mse(y_pred, y_actual):
    a=y_pred
    b=y_actual
    mse=np.square(np.subtract(a, b)).mean()
    return mse

delivery=pd.read_csv("E:\\Neeraj\\Exam and Careers\\DataScience\\Data Sets\\delivery_time.csv")
delivery
delivery.columns=("dt", "st")
delivery.corr()

delivery_normalize=preprocessing.normalize(delivery)
delivery_normalize
delivery_normalize.columns=("dtnorm","stnorm")

plt.hist(delivery.dt) #in this data outlier can be ignored
plt.hist(delivery.st) #this not following normal distribution




model=smf.ols('dt~st', data=delivery).fit()
model.summary()
predict=model.predict(delivery)
predict
plt.scatter(delivery.st, delivery.dt, color='red');plt.plot(delivery.st, predict, color='black')

mse_1=mse(predict, delivery.dt)
mse_1




model2=smf.ols('dt~np.log(st)', data=delivery).fit()
model2.summary()
predict2=model2.predict(delivery)
plt.scatter(delivery.st, delivery.dt, color='red');plt.plot(delivery.st, predict2, color='black')

mse_2=mse(predict2, delivery.dt)
mse_2                            #having least mean square error

#After doing every possible analysis this is proven to be the best model for the given dat....**solved
'''**solved'''



model3=smf.ols('np.log(dt)~st', data=delivery).fit()
model3.summary()

predict3_log=model3.predict(delivery)
predict3=np.exp(predict3_log)
predict3

plt.scatter(delivery.st, delivery.dt, color='red');plt.plot(delivery.st, predict3, color='black')

mse_3=mse(predict3, delivery.dt)
mse_3




model4=smf.ols('np.log(dt)~np.log(st)', data=delivery).fit()
model4.summary()

predict4_log=model4.predict(delivery)
predict4=np.exp(predict4_log)
plt.scatter(delivery.st, delivery.dt, color='red');plt.plot(delivery.st, predict4, color='black')

mse_4=mse(predict4, delivery.dt)
mse_4







delivery["st_sq"]=delivery.st*delivery.st
model5=smf.ols('dt~st+st_sq', data=delivery).fit()
model5.summary()

predict5=model5.predict(delivery)
predict5

plt.scatter(delivery.st, delivery.dt, color='red');plt.plot(delivery.st, predict5, color='black')
mse_5=mse(predict5, delivery.dt)
mse_5