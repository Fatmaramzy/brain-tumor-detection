# brain-tumor-detection

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # Reading data
# 

# In[2]:


df = pd.read_csv('set.csv')
df.head()


# # Count the number of rows and columns 

# In[3]:


df.shape


# # Visualize this count

# In[4]:


sns.countplot(df['Class'],label="Count")


# # data types

# In[5]:


# Look at the data types to see which columns need to be transformed / encoded to a number
df.dtypes


# In[ ]:





# # Correlation of the columns

# In[6]:


df.corr()


# # Visualize the correlation 

# In[7]:


# NOTE: To see the numbers within the cell ==>  sns.heatmap(df.corr(), annot=True)
plt.figure(figsize=(20,10))  #This is used to change the size of the figure/ heatmap
sns.heatmap(df.corr(), annot=True, fmt='.0%')
#plt.figure(figsize=(10,10)) #This is used to change the size of the figure/ heatmap
#sns.heatmap(df.iloc[:,1:12].corr(), annot=True, fmt='.0%') #Get a heap map of 11 columns, index 1-11, note index 0 is just the id column and is left out.


# # Spliting the data

# In[8]:


X = df.iloc[:, 0:8].values 
Y = df.iloc[:, 9].values 


# # Split the dataset into 75% Training set and 25% Testing set

# In[9]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)


# In[10]:


def fun(x):
    return abs(complex(x))
df["Eccentricity"]=df["Eccentricity"].apply(fun)
df.head()


# In[11]:


df.info()


# # Data Scaling

# In[ ]:





# In[12]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#X_train


# In[ ]:





# In[ ]:





# # Decision Tree Alghorithm

# In[13]:


acc_Col=[]


# In[14]:


from sklearn.metrics import accuracy_score 
def models(X_train,Y_train):
  from sklearn.tree import DecisionTreeClassifier
  tree = DecisionTreeClassifier(max_leaf_nodes=20,random_state=0)
  tree.fit(X_train, Y_train)

  print('Decision Tree Classifier Training Accuracy:', tree.score(X_train, Y_train)*100)
  return tree
model = models(X_train,Y_train)

y_predict=model.predict(X_test)
print('Decision Tree Classifier Testing Accuracy:', accuracy_score(Y_test,y_predict)*100)
acc_Col.append(accuracy_score(Y_test,y_predict)*100)


# In[ ]:





# In[ ]:





# # K Nearest Neighbor Alghorithm

# In[15]:


from sklearn.metrics import accuracy_score 
def models(X_train,Y_train):
  from sklearn.neighbors import KNeighborsClassifier
  knn = KNeighborsClassifier(n_neighbors = 5, p = 1)
  knn.fit(X_train, Y_train)

  print('K Nearest Neighbor Training Accuracy:', knn.score(X_train, Y_train)*100)
  return knn
model = models(X_train,Y_train)

y_predict=model.predict(X_test)
print('K Nearest Neighbor Testing Accuracy:', accuracy_score(Y_test,y_predict)*100)
acc_Col.append(accuracy_score(Y_test,y_predict)*100)


# # Random Forest Alghorithm

# In[16]:


def models(X_train,Y_train):
  from sklearn.ensemble import RandomForestClassifier
  forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 10)
  forest.fit(X_train, Y_train)
  print('Random Forest Classifier Training Accuracy:', forest.score(X_train, Y_train)*100)
  return forest
model = models(X_train,Y_train)

y_predict=model.predict(X_test)
print('Random Forest Classifier Testing Accuracy:', accuracy_score(Y_test,y_predict)*100)
acc_Col.append(accuracy_score(Y_test,y_predict)*100)


# 

# In[ ]:





# # Perceptron Alghorithm

# In[17]:


from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
def models(X_train,Y_train):
    
  
  model = MLPClassifier(
    max_iter=3000,
    alpha=0.1,
    activation='logistic',
    random_state=5)
  
  model.fit(X_train, Y_train)
  print('perceptron Training Accuracy:', model.score(X_train, Y_train)*100)
  
  return model
model = models(X_train,Y_train)
 

y_predict=model.predict(X_test)
print('perceptron Testing Accuracy:', accuracy_score(Y_test,y_predict)*100)
acc_Col.append(accuracy_score(Y_test,y_predict)*100)


# In[18]:


models=['DT',' KNN','RF','MLP']


# In[19]:



fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(models,acc_Col,width=0.4)
plt.xlabel('Models')
plt.ylabel("ACcuries")
plt.title('Visulaztion of Accurecy')
plt.show()

