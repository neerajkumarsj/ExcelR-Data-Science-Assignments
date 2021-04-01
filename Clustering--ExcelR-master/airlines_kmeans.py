# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 00:35:57 2020

@author: HP
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing


airlines=pd.read_csv('C:/Users/HP/Desktop/Assignment 3 (Clustering)/EastWestAirlines.csv')
airlines.drop(['ID#'], inplace=True, axis=1)
X=pd.DataFrame(preprocessing.normalize(airlines), columns=['bal', 'qual', 'cc1', 'cc2', 'cc3', 'bonusmiles', 'bonustrans', 'flightmiles', 'flighttrans', 'days', 'award' ])        


from sklearn.cluster import KMeans
wcss=[]

for i in range (1,11):
    kmeans=KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title('Elbow Plot')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

kmeans=KMeans(n_clusters=4, init='k-means++', n_init=10, max_iter=300, random_state=0)
y_kmeans=kmeans.fit(X)
kmeans.labels_
air_clust=pd.Series(kmeans.labels_)
airlines['Cluster']=air_clust
airlines=airlines.iloc[:,[11,0,1,2,3,4,5,6,7,8,9,10]]