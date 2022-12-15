# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 13:27:56 2022

@author: admin
"""

import pandas as pd
df = pd.read_csv("crime_data.csv")
df.shape
df.head()

import matplotlib.pyplot as plt
plt.scatter(df.iloc[:,3],df.iloc[:,4])
plt.show()

from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
SS_X = SS.fit_transform(df.iloc[:,2:])

#==============================================================================
# Agglomerative clustering
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 7))  
plt.title("Customer Dendograms")  
dend = shc.dendrogram(shc.linkage(SS_X, method='complete')) 

from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='complete')
Y = cluster.fit_predict(SS_X)

Y = pd.DataFrame(Y)
Y[0].value_counts()

plt.figure(figsize=(10, 7))  
plt.scatter(SS_X[:,0], SS_X[:,1], c=cluster.labels_, cmap='rainbow')  

#==============================================================================
# K-Means clustering
from sklearn.cluster import KMeans
Kmeans = KMeans(n_clusters=5,n_init=20)
Kmeans.fit(SS_X)
Y = Kmeans.predict(SS_X)

Y = pd.DataFrame(Y)
Y[0].value_counts()

plt.figure(figsize=(10, 7))  
plt.scatter(SS_X[:,0], SS_X[:,1], c=Kmeans.labels_, cmap='rainbow')  

Kmeans.inertia_
# Getting the cluster centers
C = Kmeans.cluster_centers_

l1 = []
for i in range(1,11):
    Kmeans = KMeans(n_clusters=i,n_init=20)
    Kmeans.fit(SS_X)
    l1.append(Kmeans.inertia_)
    
print(l1)

pd.DataFrame(range(1,11))        
pd.DataFrame(l1)
    
pd.concat([pd.DataFrame(range(1,11)),pd.DataFrame(l1)], axis=1)

import matplotlib.pyplot as plt
plt.scatter(range(1,11),l1)
plt.show()

from mpl_toolkits.mplot3d import Axes3D
%matplotlib qt
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(SS_X[:, 0], SS_X[:, 1], SS_X[:, 2])
ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c='Red', s=1000) 
#########################################################
# DBSCAN Clustering
from sklearn.cluster import DBSCAN
dbscan=DBSCAN(eps=1,min_samples=4)
dbscan.fit(SS_X)
dbscan.labels_
df['clusters']=dbscan.labels_
df

df.groupby('clusters').agg(['mean']).reset_index()

# Plot Clusters
plt.figure(figsize=(10, 7))  
plt.scatter(df['clusters'],df['UrbanPop'], c=dbscan.labels_)