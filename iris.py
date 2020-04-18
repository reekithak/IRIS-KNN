#!/usr/bin/env python
# coding: utf-8

# In[48]:


# Akhil Sanker - RA1811026020035 (CSE AI-ML)
import os 
import warnings
warnings.simplefilter("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score,accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# # DATA ANALYSIS 

# In[6]:


iris = pd.read_csv("iris.csv")


# In[7]:


iris.head(10)


# In[8]:


iris.tail(10)


# In[12]:


iris_data = iris.drop(['Id','Species'],axis=1)


# In[14]:


iris_data.head(10)


# In[17]:


iris_data.shape


# # Distance Calculation 

# In[20]:


wcss=[]
for x in range(1,15):
    kmeans=KMeans(n_clusters=x)
    kmeans.fit(iris_data)
    wcss.append(kmeans.inertia_)
    


# In[21]:


wcss


# In[23]:


plt.figure(figsize=(20,8))
plt.title("WCSS",fontsize=20)
plt.plot(range(1,15),wcss,'-o')
plt.grid(True)
plt.xlabel("Clusters",fontsize = 15)
plt.ylabel("Inertia",fontsize =15)
plt.xticks(range(1,15))
plt.tight_layout()
plt.show()


# In[25]:


kmeans = KMeans(n_clusters=3)
kmeans.fit(iris_data)


# In[26]:


kmeans_predict = kmeans.predict(iris_data)
kmeans_predict


# In[27]:


iris_data = pd.DataFrame(iris_data)


# In[31]:


iris_data['Label'] = kmeans_predict
iris_data.head(10)
iris_data.tail(10)


# #  Scatter plot

# In[34]:


plt.figure(figsize=(25,5))
plt.subplot(1,5,5)
plt.title("Clustered",fontsize=17)
plt.xlabel("PetalLength:cm")
plt.scatter(iris_data.SepalLengthCm[iris_data.Label ==0],iris_data.SepalWidthCm[iris_data.Label == 0])
plt.scatter(iris_data.SepalLengthCm[iris_data.Label ==1],iris_data.SepalWidthCm[iris_data.Label == 1])
plt.scatter(iris_data.SepalLengthCm[iris_data.Label ==2],iris_data.SepalWidthCm[iris_data.Label == 2])


# #  Statistical Analysis ( pairplot )

# In[36]:


sns.pairplot(data=iris_data,hue="Label",palette="Set1")
plt.show()


# In[39]:


sns.pairplot(data=iris_data,hue="Label",palette="Set3")
plt.show()


# In[ ]:


sns.pairplot(data=iris_data,hue="Label",palette="Set2")
plt.show()


# In[43]:


kmeans.score(iris_data.iloc[:,:-1])
silhouette_score(iris_data,kmeans_predict)


# 
# #  Independent Vs Dependent Variable

# In[44]:


X=iris_data.iloc[:,:-1]
Y=iris_data.iloc[:,-1]


# #  Split ! ( Test = 20% ) 
# 

# In[47]:


x_train , x_test , y_train , y_test = train_test_split(X,Y,test_size=0.2,random_state=0)


# # Train the model ( KNN ) & Test

# In[50]:


knn = KNeighborsClassifier(n_neighbors= 5)
knn.fit(x_train,y_train)


# In[53]:


y_pred = knn.predict(x_test) 
y_pred


#  # Validation | 

# In[54]:


accuracy_score(y_test,y_pred)


# In[55]:


confusion_matrix(y_test,y_pred)


# In[56]:


knn.score(x_test,y_pred)


# #  Replacing with Respective Values

# In[59]:


iris_data["Label"]=iris_data["Label"].replace({0:"Setosa",1:"Versicolor",2:"Virginica"})
#iris_data["Label"].replace({0:"Setosa",1:"Versicolor",2:"Virginica"},inplace=True)


# In[61]:


iris_data.tail(10)


# In[62]:


iris_data.head(10)


# In[ ]:




