#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


df=pd.read_csv("/kaggle/input/tunadromd-malware-detection-new/data.csv")
df.head()


# In[3]:


df.sample(5)


# In[4]:


df.isnull().sum()


# In[5]:


df=df.fillna(method="ffill",axis=0)


# In[6]:


df.isnull().sum()


# In[7]:


df.isnull().sum().max()


# In[8]:


df.dtypes


# In[9]:


df.describe()


# In[10]:


import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[11]:


sns.countplot(x=df['Label'])
plt.xticks(ticks=[0, 1], labels = ["Malware Not Detected", "Malware Detected"])
plt.show()


# In[12]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[13]:


x = df.drop('Label',axis=1)
y = df['Label']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30,random_state=101)


# In[14]:


model_params={
    'LogisticRegression':{
        'model':LogisticRegression(),
        'params':{}
    },
    'decisiontree':{
        'model':DecisionTreeClassifier(),
        'params':{
            'criterion':["gini","entropy","log_loss"],
            'splitter':["best","random"]
        }
    },
    'RandomForestClassifier':{
        'model':RandomForestClassifier(),
        'params':{
            'n_estimators':[100,200,300],
            'criterion':["gini","entropy","log_loss"]
        }
    },
    'knneighbor':{
        'model':KNeighborsClassifier(),
        'params':{
            'n_neighbors':[1,2,3,4,5,6,7,8,9,10]
        }
    }
}


# In[15]:


m=[]
for mp,model_parms in model_params.items():
    clf=GridSearchCV(model_parms['model'],model_parms['params'],cv=5)
    clf.fit(x_train,y_train)
    m.append({'params':clf.best_params_,'model':mp,'score':clf.best_score_})


# In[16]:


m


# In[17]:


dft=pd.read_csv("/kaggle/input/tunadromd-malware-detection-new/test.csv")
dft.head()


# In[18]:


dft=dft.fillna(method='ffill',axis=0)


# In[19]:


dft.head()


# In[20]:


dft.isnull().sum()


# In[21]:


X = dft.drop(columns=['ID'])
Y = dft['ID']


# In[22]:


d=pd.DataFrame(Y)


# In[23]:


ran=RandomForestClassifier()
ran.fit(x_train,y_train)


# In[24]:


d['Label']=pd.DataFrame(ran.predict(dft))


# In[25]:


d.to_csv('prediction.csv',index=False)


# In[ ]:




