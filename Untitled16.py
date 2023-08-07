#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import pandas as pd


# In[16]:


url = "https://raw.githubusercontent.com/dineshpiyasamara/LaptopPricePredictor/master/model%20building/laptop_price.csv"
data = pd.read_csv(url, encoding='latin-1')
data.head()


# In[17]:


data.shape


# In[18]:


data.isnull().sum()


# In[19]:


data.info()


# In[22]:


data.head(2)


# In[21]:


data['Ram'] = data['Ram'].str.replace('GB','').astype('int32')
data['Weight'] = data['Weight'].str.replace('kg','').astype('float32')


# In[23]:


data.corr()['Price_euros']


# In[25]:


data['Company'].value_counts()


# In[ ]:





# In[26]:


def add_company(inpt):
    if inpt == 'Samsung' or inpt == 'Razer' or inpt == 'Mediacom' or inpt == 'Microsoft'or inpt == 'Xiaomi'or inpt == 'Vero'or inpt == 'Chuwi'or inpt == 'Google'or inpt == 'Fujitsu'or inpt == 'LG'or inpt == 'Huawei':
        return 'Other'
    else:
        return inpt
    

data["Company"] = data["Company"].apply(add_company)


# In[27]:


data["Company"].value_counts()


# In[28]:


len(data['Product'].value_counts())


# In[29]:


data['TypeName'].value_counts()


# In[30]:


data['ScreenResolution'].value_counts()


# In[31]:


data['Touchscreen'] = data['ScreenResolution'].apply(lambda x:1 if 'Touchscreen' in x else 0)
data['Ips'] = data['ScreenResolution'].apply(lambda x:1 if 'IPS' in x else 0)


# In[32]:


data.head(2)


# In[33]:


data['Cpu'].value_counts()


# In[34]:


data['cpu_name'] = data['Cpu'].apply(lambda x:" ".join(x.split()[0:3]))


# In[35]:


def set_processor(name):
    if name == 'Intel Core i7' or name == 'Intel Core i5' or name == 'Intel Core i3':
        return name
    else:
        if name.split()[0] == 'AMD':
            return 'AMD'
        else:
            return 'Other'


# In[36]:


data['cpu_name'] = data['cpu_name'].apply(set_processor)


# In[37]:


data['cpu_name'].value_counts()


# In[38]:


data['Ram'].value_counts()


# In[39]:


data['Gpu'].value_counts()


# In[40]:


data['gpu_name'] = data['Gpu'].apply(lambda x:" ".join(x.split()[0:1]))


# In[41]:


data['gpu_name'].value_counts()


# In[42]:


data = data[data['gpu_name'] != 'ARM']


# In[43]:


data.shape


# In[44]:


data.head()


# In[45]:


data['OpSys'].value_counts()


# In[47]:


def set_os(inpt):
    if inpt == 'Windows 10' or inpt == 'Windows 7' or inpt == 'Windows 10 S':
        return 'Windows'
    elif inpt == 'macOS' or inpt == 'Mac OS X':
        return 'Mac'
    elif inpt == 'Linux':
        return inpt
    else:
        return 'Other'
    
data['OpSys'] = data['OpSys'].apply(set_os)


# In[48]:


data['OpSys'].value_counts()


# In[49]:


data.head()


# In[50]:


data = data.drop(columns=['laptop_ID', 'Inches', 'Product', 'ScreenResolution', 'Cpu', 'Gpu'])
data.head(5)


# In[ ]:





# In[ ]:





# In[ ]:





# In[52]:


# One-Hot Encoding


# In[53]:


data = pd.get_dummies(data)


# In[54]:



data.head()


# In[55]:


data.shape


# In[ ]:





# In[ ]:





# In[56]:


#Model Building and Selection


# In[57]:


X = data.drop('Price_euros', axis=1)
y = data['Price_euros']


# In[58]:


get_ipython().system('pip install sklearn')


# In[59]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)


# In[60]:


X_train.shape, X_test.shape


# In[64]:


def model_acc(model):
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print(str(model)+ ' --> ' +str(acc))


# In[65]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
model_acc(lr)

from sklearn.linear_model import Lasso
lasso = Lasso()
model_acc(lasso)

from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
model_acc(dt)

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
model_acc(rf)


# In[66]:


from sklearn.model_selection import GridSearchCV

parameters = {'n_estimators':[10, 50, 100],
              'criterion':['squared_error','absolute_error','poisson']}

grid_obj = GridSearchCV(estimator=rf, param_grid=parameters)

grid_fit = grid_obj.fit(X_train, y_train)

best_model = grid_fit.best_estimator_

best_model.score(X_test, y_test)


# In[67]:


import pickle
with open('predictor.pickle', 'wb') as file:
    pickle.dump(best_model, file)


# In[68]:


pred_value = best_model.predict([[8, 1.3, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0]])
pred_value


# In[ ]:




