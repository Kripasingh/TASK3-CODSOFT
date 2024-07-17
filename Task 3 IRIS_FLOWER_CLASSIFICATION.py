#!/usr/bin/env python
# coding: utf-8

# # TASK-3: IRIS FLOWER CLASSIFICATION

# IMPORTING LIBRARIES

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# IMPORTING DATASETS

# In[2]:


df= pd.read_csv("IRIS Flower.csv")


# In[3]:


df


# In[4]:


df.head()


# In[5]:


df.shape


# Data Pre-processing

# In[6]:


df.describe()


# In[7]:


df.isna().sum()


# In[8]:


df.info()


# In[9]:


#importing libraries used for encoding
from sklearn.preprocessing import LabelEncoder
from pandas.core.dtypes.common import is_numeric_dtype
le = LabelEncoder


# In[10]:


df.info()


# Data Visualization

# In[11]:


sns.countplot(x ='species', data = df, palette = "Set2")
plt.show()


# In[12]:


sns.histplot(data = df , x= df.sepal_length , color = 'blue')


# In[13]:


sns.histplot(data = df , x= df.sepal_width , color = 'purple')


# In[14]:


sns.histplot(data = df , x= df.petal_length , color = 'blue')


# In[15]:


sns.histplot(data = df , x= df.petal_width , color = 'blue')


# In[16]:


corr = df.corr()
sns.heatmap(corr, annot = True)


# Data Preparation

# In[17]:


from sklearn.model_selection import train_test_split


# In[18]:


df.columns


# In[19]:


X = df.drop(['species'], axis=1)
y = df.species


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)


# Model Building

# In[21]:


from sklearn.linear_model import LogisticRegression


# In[22]:


Model = LogisticRegression()
Model.fit(X_train,y_train)


# In[23]:


print("Score for Train data",Model.score(X_train, y_train))
print("Score for Test data",Model.score(X_test, y_test))


# In[24]:


from sklearn.ensemble import RandomForestClassifier


# In[25]:


Model2= RandomForestClassifier(n_estimators = 300 , random_state=100)
Model2.fit(X_train,y_train)


# In[26]:


print("Score for Train data",Model2.score(X_train, y_train))
print("Score for Test data",Model2.score(X_test, y_test))


# In[27]:


from sklearn.neighbors import KNeighborsClassifier


# In[28]:


Model3= KNeighborsClassifier()
Model3.fit(X_train,y_train)


# In[29]:


print("Score for Train data",Model3.score(X_train, y_train))
print("Score for Test data",Model3.score(X_test, y_test))


# Model Testing

# In[30]:


df.head(10)


# In[31]:


data = {'sepal_length': [5.2], 'sepal_width': [3.4], 'petal_length': [1.4], 'petal_width': [0.1]}
trail = pd.DataFrame(data)


# In[32]:


result = Model.predict(trail)
print("Result Species:" , result[0])

