# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 23:56:26 2020

@author: HP
"""

import scipy
from scipy import stats
import statsmodels.api as sm
import pandas as pd

cutlet = pd.read_csv("Cutlets.csv")
cutlet.columns='UnitA','UnitB'

print(stats.shapiro(cutlet.UnitA))
print(stats.shapiro(cutlet.UnitB))
#help(stats.shapiro)
#help(scipy.stats.levene)
scipy.stats.levene(cutlet.UnitA, cutlet.UnitB)

scipy.stats.ttest_ind(cutlet.UnitA, cutlet.UnitB)