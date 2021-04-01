# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 00:54:39 2020

@author: HP
"""

import pandas as pd
import scipy
from scipy import stats
import statsmodels.api as sm

Fantaloons=pd.read_csv("Faltoons.csv")
Fantaloons

malefemaleWeekdays=Fantaloons['Weekdays'].value_counts()
malefemaleWeekends=Fantaloons['Weekend'].value_counts()

Count={"Weekdays":malefemaleWeekdays, "Weekends":malefemaleWeekends}

Count_new=pd.DataFrame(Count)
Count_new

print(scipy.stats.ttest_ind(Count_new.Weekdays, Count_new.Weekends))



