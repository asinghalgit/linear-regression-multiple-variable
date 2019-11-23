#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


df = pd.read_csv("dataset.csv")


# In[4]:


df


# In[5]:


median_bedrooms = df[["bedrooms"]].agg(np.median)


# In[6]:


df.fillna(median_bedrooms,inplace=True)


# In[7]:


df


# In[8]:


from sklearn import linear_model


# In[9]:


model = linear_model.LinearRegression()


# In[10]:


model.fit(df[["area","bedrooms","age"]].values,df[["price"]].values)


# In[11]:


model.coef_


# In[12]:


model.intercept_


# In[13]:


price = model.predict(np.array([[3300,6,10]]))


# In[14]:


price


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




