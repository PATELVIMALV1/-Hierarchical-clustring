# -*- coding: utf-8 -*-
"""
Created on Tue May 19 13:13:11 2020

@author: patel
"""

"""
Perform clustering (Both hierarchical and K means clustering) for the airlines data to obtain optimum number of clusters. 
Draw the inferences from the clusters obtained.

Data Description:
 
The file EastWestAirlinescontains information on passengers who belong to an airlineâ€™s frequent flier program.
 For each passenger the data include information on their mileage history and on different ways they accrued or spent miles in the last year.
 The goal is to try to identify clusters of passengers that have similar characteristics for the purpose of targeting different segments for different types of mileage offers

ID --Unique ID

Balance--Number of miles eligible for award travel

Qual_mile--Number of miles counted as qualifying for Topflight status

cc1_miles -- Number of miles earned with freq. flyer credit card in the past 12 months:
cc2_miles -- Number of miles earned with Rewards credit card in the past 12 months:
cc3_miles -- Number of miles earned with Small Business credit card in the past 12 months:

1 = under 5,000
2 = 5,000 - 10,000
3 = 10,001 - 25,000
4 = 25,001 - 50,000
5 = over 50,000

Bonus_miles--Number of miles earned from non-flight bonus transactions in the past 12 months

Bonus_trans--Number of non-flight bonus transactions in the past 12 months

Flight_miles_12mo--Number of flight miles in the past 12 months

Flight_trans_12--Number of flight transactions in the past 12 months

Days_since_enrolled--Number of days since enrolled in flier program

Award--whether that person had award flight (free flight) or not

"""
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

data=pd.read_excel("C:\\Users\\patel\\Downloads\\EastWestAirlines.xlsx")

print(data.head)

data.dtypes

data=data.iloc[:,1:12] 

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

data['clust']=cluster_labels # creating a  new column and assigning it to new column 
#data= data.iloc[:,[7,0,1,2,3,4,5,6]]
data

# getting aggregate mean of each cluster
data.iloc[:,1:12].groupby(data.clust).median()

# creating a xlsx file 
data2=data.to_excel("EastWestAirlines.xlsx",encoding="utf-8")

data3=pd.read_excel("EastWestAirlines.xlsx") 


#############################kmeans#################### 

































