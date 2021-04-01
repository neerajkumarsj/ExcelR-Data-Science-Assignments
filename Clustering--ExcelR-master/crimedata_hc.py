# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 17:28:25 2020

@author: HP
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import linkage                        #to create dendogram based on Euclidean Distance
from sklearn.cluster import AgglomerativeClustering                #to cluster with defined cluster numbers from dendogram


crime=pd.read_csv('C:\\Users\\HP\\Desktop\\Assignment 3 (Clustering)\\crime_data.csv')


crime_norm=pd.DataFrame(preprocessing.normalize(crime.iloc[:,1:]), columns=['murder','assault','urbanpop','rape'])
crime_norm.describe()

help(linkage)
z1=linkage(crime_norm, method='single', metric='euclidean')

plt.figure(figsize=(15,5));plt.title('hierarchial dendogram of crime_data');plt.xlabel('index');plt.ylabel('Eu_distance')
sch.dendrogram(
        z1,
        leaf_rotation=0,
        leaf_font_size=8
)
plt.show()

hc_single=AgglomerativeClustering(n_clusters=4, linkage='single', affinity='euclidean').fit(crime_norm)
hc_single.labels_
cluster_labels=pd.Series(hc_single.labels_)

crime_norm['cluster']=cluster_labels
crime_norm=crime_norm.iloc[:,[4,0,1,2,3]]


z2=linkage(crime_norm, method='complete', metric='euclidean')

plt.figure(figsize=(15,8));plt.title('hierarchial dendogram 2 of crime_data');plt.xlabel('index');plt.ylabel('Eu_distance2')
sch.dendrogram(
        z2,
        leaf_rotation=0,
        leaf_font_size=8
)
plt.show()


hc_complete=AgglomerativeClustering(n_clusters=4, linkage='complete', affinity='euclidean').fit(crime_norm)
hc_complete.labels_
cluster_labels2=pd.Series(hc_complete.labels_)

crime_norm['cluster_complete']=cluster_labels2
crime_norm=crime_norm.iloc[:,[5,0,1,2,3,4]]





