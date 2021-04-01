# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 23:19:45 2020

@author: HP
"""

import pandas as pf
import scipy
from scipy import stats
import statsmodels.api as sm

CustomerOrder=pd.read_csv('Costomer+OrderForm.csv')
CustomerOrder

#count=pd.crosstab(CustomerOrder["Phillippines"], CustomerOrder["Indonesia"], CustomerOrder["Malta"], CustomerOrder["India"])
count1=CustomerOrder["Phillippines"].value_counts()     #counting the value of Discrete Categorical Data(no error and defective)
count2=CustomerOrder["Indonesia"].value_counts()        #counting the value of Discrete Categorical Data(no error and defective)
count3=CustomerOrder["Malta"].value_counts()
count4=CustomerOrder["India"].value_counts()

count={"Phillipines":count1, "Indonesia":count2, "Malta":count3, "India":count4} #making dictionary of all counts
count_new=pd.DataFrame(count)                                                     #making dataframe with that dictionary
count_new
Chisquares_results=scipy.stats.chi2_contingency(count_new)
Chisquares_results

Chisquare=[['', 'Test statistics', 'p value'], ['sample', Chisquares_results[0], Chisquares_results[1]]]  
Chisquare
Chisquares_results[0]
Chisquares_results[1]                                
#help(pd.crosstab)