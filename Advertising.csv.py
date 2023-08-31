#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[8]:


data = pd.read_csv("C:\\Users\\syama\\Downloads\\Advertising.csv")
data.head()


# In[9]:


data.shape


# In[11]:


data.info()


# In[10]:


data.isnull().sum()


# In[12]:


data.rename(columns={data.columns[0]: 'index'}, inplace=True)


# In[13]:


data.describe()


# In[15]:


data.drop(columns="index",inplace=True)


# In[16]:


# data.head()


# In[21]:


import seaborn as sns
sns.barplot(data = data)
plt.show()


# In[22]:


x = data.drop(columns="Sales",axis=1)
y = data['Sales']


# In[24]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)


# In[26]:


# x_train.shape
# y_train.shape


# In[30]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_trainscaled = sc.fit_transform(x_train)
x_testscaled = sc.fit_transform(x_test)


# In[31]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error


# In[32]:


lr=LinearRegression()
lr.fit(x_trainscaled,y_train)


# In[34]:


y_pred=lr.predict(x_testscaled)
y_pred


# In[35]:


mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("Mean Squared Error:",mse)
print("r-squared:",r2)


# In[36]:


data_pred=pd.DataFrame({"Actual":y_test,"Predicted":y_pred})
data_pred


# In[37]:


sns.regplot(x=y_test, y=y_pred, ci=None, color='red', marker="*",line_kws={"color": "green"})
plt.title("Linear Regression Actual VS Predicted values")
plt.xlabel("actual")
plt.ylabel("predict")


# In[ ]:




