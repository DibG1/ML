#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os



# In[2]:


#importing the dataset
data=pd.read_csv("Data.csv",na_values=[np.nan,'na',"NA",'n/a',"N/A","--","-"])
y=pd.read_csv("Values.csv",na_values=[np.nan,'na',"NA",'n/a',"N/A","--","-"])
data_test=pd.read_csv("Test_set_values.csv",na_values=[np.nan,'na',"NA",'n/a',"N/A","--","-"])


# In[3]:


data.head()


# In[4]:


data.head()


# In[5]:


y


# In[6]:


data=pd.merge(data, y, on = "id")


# In[7]:


#EDA:


# In[8]:


data


# In[9]:


data.info()


# In[10]:


data.describe()


# In[11]:


data.isnull().sum()


# In[12]:


for i in range(data.shape[1]):
    print(" Column Name :",data.columns[i])
    print(data.iloc[:,i].value_counts())


# In[13]:


import seaborn as sns
sns.countplot(x='status_group',data=data)


# In[14]:


for i in range(data.shape[1]):
    c=data.iloc[:,i].unique()
    print(" Column Name :",data.columns[i])
    print("Unique values :",len(c))


# In[15]:


import matplotlib.pyplot as pl
pd.crosstab(data.basin,data.status_group).plot(kind='bar',figsize=(16,6),colormap='Dark2',fontsize=10)
pl.xlabel("Geographic water basin")
pl.ylabel("count")


# In[16]:


import matplotlib.pyplot as pl
pd.crosstab(data.region,data.status_group).plot(kind='bar',figsize=(16,6),colormap='Dark2',fontsize=10)
pl.xlabel("Region")
pl.ylabel("count")


# In[17]:


import matplotlib.pyplot as pl
pd.crosstab(data.region_code,data.status_group).plot(kind='bar',figsize=(16,6),colormap='Dark2',fontsize=10)
pl.xlabel("Region_Code")
pl.ylabel("count")


# In[18]:


import matplotlib.pyplot as pl
pd.crosstab(data.district_code,data.status_group).plot(kind='bar',figsize=(16,6),colormap='Dark2',fontsize=10)
pl.xlabel("District_Code")
pl.ylabel("count")


# In[19]:


import matplotlib.pyplot as pl
pd.crosstab(data.public_meeting,data.status_group).plot(kind='bar',figsize=(16,6),colormap='Dark2',fontsize=10)
pl.xlabel("Public Meeting")
pl.ylabel("count")


# In[20]:


import matplotlib.pyplot as pl
pd.crosstab(data.scheme_management,data.status_group).plot(kind='bar',figsize=(16,6),colormap='Dark2',fontsize=10)
pl.xlabel("Scheme Management")
pl.ylabel("count")


# In[21]:


import matplotlib.pyplot as pl
pd.crosstab(data.permit,data.status_group).plot(kind='bar',figsize=(16,6),colormap='Dark2',fontsize=10)
pl.xlabel("Permit")
pl.ylabel("count")


# In[22]:


#Data Preprocessing


# In[23]:


data.drop(labels=['id','date_recorded','funder','installer','wpt_name','subvillage','lga','ward','recorded_by','scheme_name'],inplace=True,axis='columns')


# In[24]:


data.shape


# In[25]:


data.info()


# In[26]:


data.isnull().sum()


# In[27]:


data.drop(labels=['scheme_management'],inplace=True,axis='columns')


# In[28]:


mean=data['public_meeting'].mean()
data['public_meeting'].fillna(mean,inplace=True)


# In[29]:


mean=data['permit'].mean()
data['permit'].fillna(mean,inplace=True)


# In[30]:


data.columns


# In[31]:


data['status_group'].value_counts()


# In[32]:


data['status_group']=data['status_group'].replace({'functional':3,'functional needs repair':2,'non functional':1})


# In[33]:


data['status_group'].value_counts()


# In[34]:


data.corr().abs()


# In[35]:



def heatmap(x, y, size):
    fig, ax = pl.subplots()
    
    # Mapping from column names to integer coordinates
    x_labels = [v for v in sorted(x.unique())]
    y_labels = [v for v in sorted(y.unique())]
    x_to_num = {p[1]:p[0] for p in enumerate(x_labels)} 
    y_to_num = {p[1]:p[0] for p in enumerate(y_labels)} 
    
    size_scale = 500
    ax.scatter(
        x=x.map(x_to_num), # Use mapping for x
        y=y.map(y_to_num), # Use mapping for y
        s=size * size_scale, # Vector of square sizes, proportional to size parameter
        marker='s' # Use square as scatterplot marker
    )
    
    # Show column labels on the axes
    ax.set_xticks([x_to_num[v] for v in x_labels])
    ax.set_xticklabels(x_labels, rotation=45, horizontalalignment='right')
    ax.set_yticks([y_to_num[v] for v in y_labels])
    ax.set_yticklabels(y_labels)
    
data = data
columns = data.columns 
corr = data[columns].corr()
corr = pd.melt(corr.reset_index(), id_vars='index') # Unpivot the dataframe, so we can get pair of arrays for x and y
corr.columns = ['x', 'y', 'value']
heatmap(
    x=corr['x'],
    y=corr['y'],
    size=corr['value'].abs()
)


# In[36]:


dummydata=data


# In[37]:


dummydata=dummydata.drop(labels=['construction_year','gps_height','district_code','region_code',],axis='columns')


# In[38]:


data.columns


# In[ ]:





# In[ ]:





# In[39]:


dummydata.shape


# In[40]:


data=dummydata


# In[41]:


data


# In[42]:


data.std()


# In[43]:


data_iqr=data
Q1=data_iqr.quantile(0.25)
Q3=data_iqr.quantile(0.75)
IQR=Q3-Q1


# In[44]:


data_iqr


# In[45]:


#Outlier treatment
sorted(data_iqr)
db=data_iqr
clean_data=db[~(((db<(Q1-1.5*IQR))|(db>(Q3+1.5*IQR))).any(axis=1))]


# In[46]:


clean_data


# In[47]:


clean_data.std()


# In[48]:


data=clean_data


# In[ ]:





# In[49]:


data.dtypes


# In[50]:


data=data.drop(labels=['extraction_type_group', 'extraction_type_class','management_group','payment_type','quality_group','quantity_group','source_type','source_class','waterpoint_type_group'],axis='columns')


# In[51]:


data2=pd.get_dummies(data)


# In[52]:


data=data2


# In[53]:


X=data.iloc[:,:-1]
y=data.iloc[:,-1]
import statsmodels.api as sm


# In[54]:


X.shape


# In[ ]:





# In[55]:


y.shape


# In[56]:


#splitting


# In[57]:


from sklearn.model_selection import train_test_split


# In[58]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=1)


# In[59]:


#feature scaling


# In[60]:


from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)


# In[61]:


from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
clf = OneVsRestClassifier(LogisticRegression(random_state=1)).fit(X_train, y_train)


# In[62]:


#Predicting the test set results
y_pred=clf.predict(X_test)


# In[63]:


#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)


# In[64]:


cm


# In[85]:


clf.score(X_test,y_test)


# In[84]:


clf.score(X_test,y_pred)


# In[70]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[ ]:


#data['status_group']=data['status_group'].replace({'functional':3,'functional needs repair':2,'non functional':1})


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




