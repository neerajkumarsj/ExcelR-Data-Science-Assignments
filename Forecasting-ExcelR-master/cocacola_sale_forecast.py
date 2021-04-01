# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 14:40:18 2020

@author: HP
"""

import numpy as np
import pandas as pd
cocacola = pd.read_csv("D:\\Data Analytics Assignments\\Forecasting\\CocaCola_Sales_Rawdata.csv")
cocacola.Sales.plot()

cocacola['t_sq']=cocacola.t*cocacola.t
cocacola['log_sales']=np.log(cocacola.Sales)



cocacola['quarters']=0
p=cocacola['Quarter'][0]
for i in range(42):
    p=cocacola['Quarter'][i]
    cocacola['quarters'][i]=p[0:2]
quarter_dummies=pd.DataFrame(pd.get_dummies(cocacola['quarters']))



cocacola['years']=0
p=cocacola['Quarter'][0]   
for i in range(42):
    p=cocacola['Quarter'][i]
    cocacola['years'][i]='year' + p[2:]
year_dummies=pd.DataFrame(pd.get_dummies(cocacola['years']))

cocacola=pd.concat([cocacola, quarter_dummies, year_dummies], axis=1)

Train = cocacola.head(36)
Test = cocacola.tail(6)
cocacola.Q1.plot();cocacola.Sales.plot()


import statsmodels.formula.api as smf


#model_based_approach

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


add_season_model = smf.ols('Sales~Q1+Q2+Q3+Q4',data=Train).fit()
pred_add_season =  pd.Series(add_season_model.predict(pd.DataFrame(Test[['Q1','Q2','Q3','Q4']])))
rmse_add_season = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_season))**2))
rmse_add_season


add_season_quad_model=smf.ols('Sales~t+t_sq+Q1+Q2+Q3+Q4',data=Train).fit()
pred_add_season_quad=pd.Series(add_season_quad_model.predict(pd.DataFrame(Test[['Q1','Q2','Q3','Q4','t','t_sq']])))
rmse_add_season_quad=np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_season_quad))**2))
rmse_add_season_quad


mult_season_model=smf.ols('log_sales~Q1+Q2+Q3+Q4',data=Train).fit()
pred_mult_season=pd.Series(mult_season_model.predict(pd.DataFrame(Test[['Q1','Q2','Q3','Q4']])))
rmse_mult_season=np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_mult_season)))**2))
rmse_mult_season



mult_add_season_model=smf.ols('log_sales~Q1+Q2+Q3+Q4+t',data=Train).fit()
pred_mult_add_season=pd.Series(mult_add_season_model.predict(pd.DataFrame(Test[['Q1','Q2','Q3','Q4','t']])))
rmse_mult_add_season=np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_mult_add_season)))**2))
rmse_mult_add_season



mult_quad_season_model=smf.ols('log_sales~Q1+Q2+Q3+Q4+t+t_sq',data=Train).fit()
pred_mult_quad_season=pd.Series(mult_quad_season_model.predict(pd.DataFrame(Test[['Q1','Q2','Q3','Q4','t','t_sq']])))
rmse_mult_quad_season=np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_mult_quad_season)))**2))
rmse_mult_quad_season



add_season_quad_yearly_model=smf.ols('Sales~t+t_sq+Q1+Q2+Q3+Q4+year_86+year_87+year_88+year_89+year_90+year_91+year_92+year_93+year_94+year_95+year_96',data=Train).fit()
pred_add_season_yearly_quad=pd.Series(add_season_quad_yearly_model.predict(pd.DataFrame(Test[['Q1','Q2','Q3','Q4','year_86','year_87','year_88','year_89','year_90','year_91','year_92','year_93','year_94','year_95','year_96','t','t_sq']])))
rmse_add_season_yearly_quad=np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_season_yearly_quad))**2))
rmse_add_season_yearly_quad


model_data={'Model':pd.Series(['rmse_linear','rmse_quad','rmse_exp','rmse_add_season','rmse_add_season_quad','rmse_mult_season','rmse_mult_add_season','rmse_mult_quad_season','rmse_add_season_yearly_quad']), 'RMSE_values':pd.Series([rmse_linear,rmse_quad,rmse_exp,rmse_add_season,rmse_add_season_quad,rmse_mult_season,rmse_mult_add_season,rmse_mult_quad_season,rmse_add_season_yearly_quad])}
table_rmse=pd.DataFrame(model_data)
table_rmse



###### According to RMSE table add_season_quad_yearly_model gives the least rmse value. So this is the best model......




























