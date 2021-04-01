# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 00:33:11 2020

@author: HP
"""

import pandas as pd
import scipy
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols


labtat=pd.read_csv('LabTAT.csv')
labtat
labtat.columns='Lab1','Lab2','Lab3','Lab4'
print(stats.shapiro(labtat.Lab1))
print(stats.shapiro(labtat.Lab2))
print(stats.shapiro(labtat.Lab3))
print(stats.shapiro(labtat.Lab4))

scipy.stats.levene(labtat.Lab1, labtat.Lab2, labtat.Lab3, labtat.Lab4)
#scipy.stats.levene(labtat.Lab2, labtat.Lab3)
#scipy.stats.levene(labtat.Lab3, labtat.Lab4)
#scipy.stats.levene(labtat.Lab4, labtat.Lab1)
#scipy.stats.levene(labtat.Lab2, labtat.Lab4)
#cipy.stats.levene(labtat.Lab1, labtat.Lab3)

mod=ols('Lab1~Lab2+Lab3+Lab4', data=labtat).fit()
aov_table=sm.stats.anova_lm(mod, type=2)
print(aov_table)
