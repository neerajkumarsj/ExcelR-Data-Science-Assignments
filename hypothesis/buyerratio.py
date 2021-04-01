# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 13:29:43 2020

@author: HP
"""

import pandas as pd
import scipy
from scipy import stats
import statsmodels.api as sm

buyer=pd.read_csv('BuyerRatio.csv')
buyer
buyerdata=buyer.iloc[0:2, 1:5]
buyerdata
chisquare_results=scipy.stats.chi2_contingency(buyerdata)
Chi_square=[['','Test Statistic','p-value'],['Sample Data',chisquare_results[0],chisquare_results[1]]]
Chi_square