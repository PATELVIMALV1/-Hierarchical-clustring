# -*- coding: utf-8 -*-
"""
Created on Wed May 20 14:23:24 2020

@author: patel
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv("C:\\Users\\patel\\Downloads\\crime_data.csv") 
data.head()

data=data.iloc[:,1:]

from sklearn.preprocessing import MinMaxScaler
norm=MinMaxScaler()  
norm.fit(data)
norm_data=norm.transform(data) 

type(norm_data)

from scipy.cluster.hierarchy import linkage 

import scipy.cluster.hierarchy as sch #dendrogram 
help(linkage)

z = linkage(norm_data, method="complete",metric="euclidean")

plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram( 
    z,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()

# Now applying AgglomerativeClustering choosing 3 as clusters from the dendrogram
from	sklearn.cluster	import	AgglomerativeClustering 
h_complete	=	AgglomerativeClustering(n_clusters=3,	linkage='complete',affinity = "euclidean").fit(norm_data) 


cluster_labels=pd.Series(h_complete.labels_)

data=pd.read_csv("C:\\Users\\patel\\Downloads\\crime_data.csv")

data['clust']=cluster_labels # creating a  new column and assigning it to new column 
#data= data.iloc[:,[7,0,1,2,3,4,5,6]]
data

# getting aggregate mean of each cluster
data.iloc[:,1:].groupby(data.clust).median()

# creating a csv file 
data2=data.to_csv("crime_data.csv",encoding="utf-8")

data3=pd.read_csv("crime_data.csv") 





