# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 19:47:53 2020

@author: HP
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sc_hier
from scipy.cluster.hierarchy import linkage

airlines=pd.read_csv('C:\\Users\\HP\\Desktop\\Assignment 3 (Clustering)\\EastWestAirlines.csv')
airlines.drop(['ID#'], inplace=True, axis=1)

airlines_norm=pd.DataFrame(preprocessing.normalize(airlines), columns=['bal', 'qual', 'cc1', 'cc2', 'cc3', 'bonusmiles', 'bonustrans', 'flightmiles', 'flighttrans', 'days', 'award' ])        
z1=linkage(airlines_norm, method='average', metric='euclidean')
z1    

plt.figure(figsize=(10,15));plt.title('Dendogram of airlines data');plt.xlabel('index');plt.ylabel('eu_distance')

sc_hier.dendrogram(
        z1,
        leaf_font_size=8,
        leaf_rotation=0
)
plt.show()

hc_ave=AgglomerativeClustering(n_clusters=6, linkage='average', affinity='euclidean').fit(airlines_norm)
hc_ave.labels_
hc_label=pd.Series(hc_ave.labels_)

airlines_norm['cluster_ave']=hc_label
airlines_norm=airlines_norm.iloc[:,[11,0,1,2,3,4,5,6,7,8,9,10]]


#airlines_norm=pd.DataFrame(preprocessing.normalize(airlines), columns=['bal', 'qual', 'cc1', 'cc2', 'cc3', 'bonusmiles', 'bonustrans', 'flightmiles', 'flighttrans', 'days', 'award' ])        
z2=linkage(airlines_norm, method='complete', metric='euclidean')
z2 

plt.figure(figsize=(10,15));plt.title('Dendogram of airlines data');plt.xlabel('index');plt.ylabel('eu_distance')

sc_hier.dendrogram(
        z2,
        leaf_font_size=8,
        leaf_rotation=0
)
plt.show()

hc_complete=AgglomerativeClustering(n_clusters=5, linkage='complete', affinity='euclidean').fit(airlines_norm)
hc_complete.labels_
hc_label2=pd.Series(hc_complete.labels_)

airlines_norm['cluster_complete']=hc_label2
airlines_norm=airlines_norm.iloc[:,[12,0,1,2,3,4,5,6,7,8,9,10,11]]

z3=linkage(airlines_norm, method='single', metric='euclidean')
z3

plt.figure(figsize=(10,15));plt.title('Dendogram of airlines data');plt.xlabel('index');plt.ylabel('eu_distance')

sc_hier.dendrogram(
        z3,
        leaf_font_size=8,
        leaf_rotation=0
)
plt.show()

hc_single=AgglomerativeClustering(n_clusters=5, linkage='complete', affinity='euclidean').fit(airlines_norm)
hc_single.labels_
hc_label3=pd.Series(hc_single.labels_)

airlines_norm['cluster_single']=hc_label3
airlines_norm=airlines_norm.iloc[:,[13,0,1,2,3,4,5,6,7,8,9,10,11,12]]


