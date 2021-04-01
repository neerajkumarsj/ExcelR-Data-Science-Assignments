# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

dataset=pd.read_csv('C:/Users/HP/Desktop/Assignment 3 (Clustering)/crime_data.csv')

X=dataset.iloc[:, 1:].values
X

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
crime_clust=pd.Series(kmeans.labels_)
dataset['Cluster']=crime_clust
dataset=dataset.iloc[:,[5,0,1,2,3,4]]


