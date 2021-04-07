import pandas as pd
import numpy as np
import os
data=pd.read_csv("creditcard.csv",na_values=[np.nan,'na',"NA",'n/a',"N/A","--","-"])
data
data.info()
data.describe()
data.isnull().sum()
for i in range(31):
  mean=data.iloc[:,i].mean()
  data.iloc[:,i].fillna(mean,inplace=True)
data['Class'].value_counts()
data.isnull().sum()
import seaborn as sns
sns.countplot(data['Class'])
for i in range(data.shape[1]):
    c=data.iloc[:,i].unique()
    print(" Column Name :",data.columns[i])
    print("Unique values :",len(c))
data.corr().abs()
import matplotlib.pyplot as pl
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
xdata = data.iloc[:,:-1]
columns = xdata.columns 
corr = xdata[columns].corr()
corr = pd.melt(corr.reset_index(), id_vars='index') # Unpivot the dataframe, so we can get pair of arrays for x and y
corr.columns = ['x', 'y', 'value']
heatmap(
    x=corr['x'],
    y=corr['y'],
    size=corr['value'].abs()
)
data
data.std()
X=data.iloc[:,:-1]
Q1=X.quantile(0.25)
Q3=X.quantile(0.75)
IQR=Q3-Q1
#Outlier treatment
sorted(X)
db=X
clean_data=db[~(((db<(Q1-1.5*IQR))|(db>(Q3+1.5*IQR))).any(axis=1))]
X.std()
y=data.iloc[:,-1].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=1,stratify=y)
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)
#fitting SVC classifier to the training set 
from sklearn.svm import SVC
classifier=SVC(kernel='rbf',random_state=1)
classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)
#CM
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm
classifier.score(X_test,y_test)
classifier.score(X_test,y_pred)
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
#Fitting MLPClassifier
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
y1_pred=clf.predict(X_test)
cm=confusion_matrix(y_test,y1_pred)
cm
clf.score(X_test,y1_pred)
clf.score(X_test,y_test)
print(classification_report(y_test,y1_pred))
print("Total number of fake transactions")
print(np.count_nonzero(y_test == 1))
print("Fake transactions detected by SVM")
print(np.count_nonzero(y_pred == 1))
print("Fake transactions detected by MLP")
print(np.count_nonzero(y1_pred == 1))

