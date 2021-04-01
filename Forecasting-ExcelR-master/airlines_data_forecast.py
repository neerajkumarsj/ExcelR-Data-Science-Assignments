# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 13:15:08 2020

@author: HP
"""

import numpy as np
import pandas as pd
# sklearn.preprocessing 
airlines = pd.read_csv("D:\\Data Analytics Assignments\\Forecasting\\Airlines+Data.csv")
airlines.Passengers.plot()
airlines['t_sq']=airlines.t*airlines.t
airlines['log_passenger']=np.log(airlines.Passengers)
month =['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'] 

import numpy as np

################# Dummy for months ####################################

p = airlines["Month"][0]
p[0:3]
airlines['months']= 0

for i in range(96):
    p = airlines["Month"][i]
    airlines['months'][i]= p[0:3]    
month_dummies = pd.DataFrame(pd.get_dummies(airlines['months']))


################### Dummy for years ############################


p = airlines['Month'][0]
p[3:6]
airlines['years']=0
for i in range(96):
    p=airlines['Month'][i]
    airlines['years'][i]='year ' + p[3:6]
year_dummies = pd.DataFrame(pd.get_dummies(airlines['years']))

airlines=pd.concat([airlines, month_dummies, year_dummies], axis=1)


Train = airlines.head(80)
Test = airlines.tail(16)

import statsmodels.formula.api as smf 



#model_based_approach

linear_model = smf.ols('Passengers~t',data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_linear))**2))
rmse_linear

quad_model = smf.ols('Passengers~t+t_sq',data=Train).fit()
pred_quad =  pd.Series(quad_model.predict(pd.DataFrame(Test[['t','t_sq']])))
rmse_quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_quad))**2))
rmse_quad

exp_model = smf.ols('log_passenger~t',data=Train).fit()
pred_exp =  pd.Series(exp_model.predict(pd.DataFrame(Test[['t']])))
rmse_exp = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_exp)))**2))
rmse_exp


add_season_model = smf.ols('Passengers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec',data=Train).fit()
pred_add_season =  pd.Series(add_season_model.predict(pd.DataFrame(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']])))
rmse_add_season = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_season))**2))
rmse_add_season


add_season_quad_model=smf.ols('Passengers~t+t_sq+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec',data=Train).fit()
pred_add_season_quad=pd.Series(add_season_quad_model.predict(pd.DataFrame(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','t','t_sq']])))
rmse_add_season_quad=np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_season_quad))**2))
rmse_add_season_quad


mult_season_model=smf.ols('log_passenger~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec',data=Train).fit()
pred_mult_season=pd.Series(mult_season_model.predict(pd.DataFrame(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']])))
rmse_mult_season=np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_mult_season)))**2))
rmse_mult_season



mult_add_season_model=smf.ols('log_passenger~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec+t',data=Train).fit()
pred_mult_add_season=pd.Series(mult_add_season_model.predict(pd.DataFrame(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','t']])))
rmse_mult_add_season=np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_mult_add_season)))**2))
rmse_mult_add_season



mult_quad_season_model=smf.ols('log_passenger~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec+t+t_sq',data=Train).fit()
pred_mult_quad_season=pd.Series(mult_quad_season_model.predict(pd.DataFrame(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','t','t_sq']])))
rmse_mult_quad_season=np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_mult_quad_season)))**2))
rmse_mult_quad_season



model_data={'Model':pd.Series(['rmse_linear','rmse_quad','rmse_exp','rmse_add_season','rmse_add_season_quad','rmse_mult_season','rmse_mult_add_season','rmse_mult_quad_season']), 'RMSE_values':pd.Series([rmse_linear,rmse_quad,rmse_exp,rmse_add_season,rmse_add_season_quad,rmse_mult_season,rmse_mult_add_season,rmse_mult_quad_season])}
table_rmse=pd.DataFrame(model_data)
table_rmse

#20 dummy variables created...... 8 for 8 years and 12 months
#mult_add_season model is the best model to predict passenger of a given month in a year



