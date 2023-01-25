#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# In[2]:


import os
os.environ['OMP_NUM_THREADS'] = '1'


# In[3]:


# importing the dataset
dataset = pd.read_csv('wholesale_customers.csv')


# In[6]:


dataset.head()


# In[5]:


dataset.info()


# In[7]:


dataset.describe()


# In[8]:


sns.pairplot(dataset.iloc[:,[2,3,4]])


# In[9]:


from sklearn.preprocessing import StandardScaler
x = dataset.iloc[:,[3,4]].values
sc_x = StandardScaler()
x = sc_x.fit_transform(x)


# In[10]:


# using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 7):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 30)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 7), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('wcss')
plt.show()


# In[11]:


# fitting k-means to the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(x)


# In[13]:


# visualizing the clusters
plt.figure(figsize = (8,8))
plt.scatter(x[y_kmeans == 0,0],x[y_kmeans == 0,1], s = 100, c = 'red', label ='cluster 1')
plt.scatter(x[y_kmeans == 1,0],x[y_kmeans == 1,1], s = 100, c = 'blue', label = 'cluster 2')
plt.scatter(x[y_kmeans == 2,0],x[y_kmeans == 2,1], s = 100, c = 'green', label = 'cluster 3')
plt.scatter(x[y_kmeans == 3,0],x[y_kmeans == 3,1], s = 100, c = 'cyan',  label = 'cluster 4')
plt.scatter(x[y_kmeans == 4,0],x[y_kmeans == 4,1], s = 100, c = 'yellow', label = 'cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c ='yellow', label = 'centroids')
plt.title('clusters of customers')
plt.xlabel('Annual Income (Scaled)')
plt.ylabel('spending score(scaled)')
plt.legend()
plt.show()


# In[17]:


# using the dendogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch

plt.figure(figsize = (15, 6))
dendogram = sch.dendrogram(sch.linkage(x, method = 'ward'))
plt.title('Dendogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()


# In[19]:


# fitting Hierarchical clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 4, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(x)


# In[21]:


# visualizing the clusters
plt.figure(figsize = (6,6))
plt.scatter(x[y_hc == 0,0], x[y_hc == 0, 1], s = 100, c = 'red', label = 'cluster 1')
plt.scatter(x[y_hc == 1,0], x[y_hc == 1, 1], s = 100, c = 'blue', label = 'cluster 2')
plt.scatter(x[y_hc == 2,0], x[y_hc == 2, 1], s = 100, c = 'green', label = 'cluster 3')
plt.scatter(x[y_hc == 3,0], x[y_hc == 3, 1], s = 100, c = 'cyan', label = 'cluster 4')
plt.scatter(x[y_hc == 4,0], x[y_hc == 4, 1], s = 100, c = 'yellow', label = 'cluster 5')
plt.title('clusters of customers')
plt.xlabel('Annual Income (scaled)')
plt.ylabel('spending score (scaled)')
plt.legend()
plt.show()


# In[22]:


from sklearn.neighbors import NearestNeighbors

neighbours = NearestNeighbors(n_neighbors = 2)
distances, indices = neighbours.fit(x).kneighbors(x)

distances = distances[:, 1]
distances = np.sort(distances, axis = 0)
plt.plot(distances)


# In[25]:


from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps = 0.25, min_samples = 5)
y_dbscan = dbscan.fit_predict(x)


# In[26]:


# inspect the array to identify number of clusters

y_dbscan


# In[28]:


# visualizing the clusters
plt.figure(figsize = (6, 6))
plt.scatter(x[y_dbscan == 0, 0], x[y_dbscan == 0, 1], s = 100, c = 'red', label = 'cluster 1')
plt.scatter(x[y_dbscan == 1, 0], x[y_dbscan == 1, 1], s = 100, c = 'blue', label = 'cluster 2')
plt.scatter(x[y_dbscan == 2, 0], x[y_dbscan == 2, 1], s = 100, c = 'green', label = 'cluster 3')
plt.scatter(x[y_dbscan == 3, 0], x[y_dbscan == 3, 1], s = 100, c = 'cyan', label = 'cluster 4')
plt.scatter(x[y_dbscan == 4, 0], x[y_dbscan == 4, 1], s = 100, c = 'yellow', label = 'cluster 5')
plt.scatter(x[y_dbscan == 5, 0], x[y_dbscan == 5, 1], s = 100, c = 'brown', label = 'cluster 6')
plt.scatter(x[y_dbscan == -1, 0], x[y_dbscan == -1, 1], s = 100, c = 'black', label = 'Noise')
plt.title('clusters of customers')
plt.xlabel('Annual Income (scaled)')
plt.ylabel('Spending Score (scaled)')
plt.legend()
plt.show()


# In[29]:


financials = pd.read_csv('costpercompany.csv')
financials.info()


# In[30]:


financials.describe()


# In[31]:


financials.head()


# In[33]:


x = financials.iloc[:,1:6]
sns.pairplot(x)


# In[34]:


# scale the data

sc_x = StandardScaler()
x = sc_x.fit_transform(x)


# In[36]:


# using the elbow method to find the optimal number of clusters

wcss =[]
for i in range (1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 30)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[37]:


# fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 30)
y_kmeans = kmeans.fit_predict(x)


# In[39]:


# we need to reduce dimensionality before we can visualise

from sklearn.decomposition import PCA

pca = PCA(n_components = 2)
x_reduced = pca.fit_transform(x)

pca.explained_variance_ratio_


# In[40]:


sum(pca.explained_variance_ratio_)


# In[1]:


# visualizing the clusters

colours = ['red', 'blue', 'green']

plt.figure(figsize = (6,6))
for i in range(4):
    plt.scatter(x_reduced[y_kmeans == i, 0], x_reduced[y_kmeans == i,1],
                s = 100, c = colours[1], label = 'cluster' +str(i+1))
plt.title('clusters of companies')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.legend()
plt.show()


# In[ ]:




