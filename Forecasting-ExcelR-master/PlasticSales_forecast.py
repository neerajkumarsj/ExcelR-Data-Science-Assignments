# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 18:35:45 2020

@author: HP
"""

import numpy as np
import pandas as pd
plasticsale = pd.read_csv("D:\\Data Analytics Assignments\\Forecasting\\PlasticSales.csv")
plasticsale.Sales.plot()
from sklearn.preprocessing import scale

plasticsale['t']=0
for i in range(0,60):
    plasticsale['t'][i]=i+1


plasticsale['t_sq']=plasticsale.t*plasticsale.t
plasticsale['log_sales']=np.log(plasticsale.Sales)

plasticsale.plot()

#plasticsale[['Sales','t','t_sq','log_sales']]=pd.DataFrame(scale(plasticsale.iloc[:,1:]))
#plasticsale.plot()


################# Dummy for months ####################################

p = plasticsale["Month"][0]
p[0:3]
plasticsale['months']= 0

for i in range(60):
    p = plasticsale["Month"][i]
    plasticsale['months'][i]= p[0:3]    
month_dummies = pd.DataFrame(pd.get_dummies(plasticsale['months']))


################### Dummy for years ############################


p = plasticsale['Month'][0]
p[3:6]
plasticsale['years']=0
for i in range(60):
    p=plasticsale['Month'][i]
    plasticsale['years'][i]='year' + p[3:]
year_dummies = pd.DataFrame(pd.get_dummies(plasticsale['years']))

plasticsale=pd.concat([plasticsale, month_dummies, year_dummies], axis=1)

######################################### model building ############################################

Train=plasticsale.head(50)
Test=plasticsale.tail(10)


import statsmodels.formula.api as smf



linear_model = smf.ols('Sales~t',data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_linear))**2))
rmse_linear

quad_model = smf.ols('Sales~t+t_sq',data=Train).fit()
pred_quad =  pd.Series(quad_model.predict(pd.DataFrame(Test[['t','t_sq']])))
rmse_quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_quad))**2))
rmse_quad

exp_model = smf.ols('log_sales~t',data=Train).fit()
pred_exp =  pd.Series(exp_model.predict(pd.DataFrame(Test[['t']])))
rmse_exp = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_exp)))**2))
rmse_exp


add_season_model = smf.ols('Sales~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec',data=Train).fit()
pred_add_season =  pd.Series(add_season_model.predict(pd.DataFrame(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']])))
rmse_add_season = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_season))**2))
rmse_add_season


add_season_quad_model=smf.ols('Sales~t+t_sq+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec',data=Train).fit()
pred_add_season_quad=pd.Series(add_season_quad_model.predict(pd.DataFrame(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','t','t_sq']])))
rmse_add_season_quad=np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_season_quad))**2))
rmse_add_season_quad


mult_season_model=smf.ols('log_sales~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec',data=Train).fit()
pred_mult_season=pd.Series(mult_season_model.predict(pd.DataFrame(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']])))
rmse_mult_season=np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_mult_season)))**2))
rmse_mult_season



mult_add_season_model=smf.ols('log_sales~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec+t',data=Train).fit()
pred_mult_add_season=pd.Series(mult_add_season_model.predict(pd.DataFrame(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','t']])))
rmse_mult_add_season=np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_mult_add_season)))**2))
rmse_mult_add_season



mult_quad_season_model=smf.ols('log_sales~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec+t+t_sq',data=Train).fit()
pred_mult_quad_season=pd.Series(mult_quad_season_model.predict(pd.DataFrame(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','t','t_sq']])))
rmse_mult_quad_season=np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_mult_quad_season)))**2))
rmse_mult_quad_season

model_data={'Model':pd.Series(['rmse_linear','rmse_quad','rmse_exp','rmse_add_season','rmse_add_season_quad','rmse_mult_season','rmse_mult_add_season','rmse_mult_quad_season']), 'RMSE_values':pd.Series([rmse_linear,rmse_quad,rmse_exp,rmse_add_season,rmse_add_season_quad,rmse_mult_season,rmse_mult_add_season,rmse_mult_quad_season])}
table_rmse=pd.DataFrame(model_data)
table_rmse


### from table_rmse we can see rmse_mult_add_season is th least root mean square.
### mult_add_season_model is the best model






































