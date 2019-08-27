#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.datasets import load_iris
import seaborn as sns; sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


iris = load_iris()


# In[8]:


df = pd.DataFrame(iris.data, columns=iris.feature_names)


# In[9]:


df['species'] = pd.Series(iris.target)


# In[10]:


df.info()


# In[ ]:





# In[ ]:





# In[11]:


sl_df = pd.DataFrame()


# In[14]:


sl_df['sepal_length'] = df['sepal length (cm)']
sl_df['species'] = df['species']
sl_df.info()


# In[15]:


sl_df = sl_df[:100]
sl_df.info()


# In[16]:


sl_df.describe()


# In[19]:


sns.pairplot(sl_df, hue='species')


# In[20]:


# 학습 데이터, 테스트 데이터 나누기
from sklearn.model_selection import train_test_split


# In[25]:


X_train, X_test, y_train, y_test = train_test_split( sl_df.iloc[:,:1], 
                                                    sl_df.iloc[:, 1:], test_size=0.33)

print(X_trian)
# In[ ]:





# In[26]:


from sklearn.linear_model import LogisticRegression


# In[28]:


lr = LogisticRegression()


# In[30]:


lr.fit(X_train, y_train)


# In[41]:


import seaborn as sns; sns.set()
sns.lmplot(x='sepal_length', y='species', data=sl_df, logistic=True)


# In[37]:


from sklearn.metrics import classification_report, confusion_matrix


# In[42]:


print(confusion_matrix(y_train, lr.predict(X_train)))


# In[43]:


print(classification_report(y_train, lr.predict(X_train)))


# In[44]:


print(confusion_matrix(y_test, lr.predict(X_test)))


# In[45]:


print(classification_report(y_test, lr.predict(X_test)))


# In[ ]:





# In[47]:


import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')


# In[48]:


iris = load_iris()


# In[53]:


df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Series(iris.target)


# In[56]:


def setcolor(value):
    color = []
    colors = ['r', 'g', 'b']
    for i in value.values:
        color.append(colors[i])
    return color


# In[58]:


plt.scatter(x=df['petal length (cm)'], y=df['petal width (cm)'], 
            color = setcolor(df['species']))


# In[59]:


from sklearn.neighbors import KNeighborsClassifier


# In[62]:


column_train = ['petal length (cm)', 'petal width (cm)']
neigh_3 = KNeighborsClassifier(n_neighbors = 3, weights = 'distance')


# In[63]:


neigh_3_train = neigh_3.fit(df[column_train], df['species'])


# In[65]:


new_data = np.array([2.5, 0.8]).reshape(1,-1)


# In[70]:


# 학습 자료 출력
plt.scatter(x=df['petal length (cm)'], y=df['petal width (cm)'], 
           color=setcolor(df['species']))
# 새 자료의 위치
plt.scatter(x=new_data[0,0], y=new_data[0,1], color='orange')


# In[74]:


neigh_3_class = neigh_3_train.predict(new_data)


# In[75]:


print(neigh_3_class)


# In[128]:


# 학습 자료 출력
plt.scatter(x=df['petal length (cm)'], y=df['petal width (cm)'], 
           color=setcolor(df['species']))

# 새 자료의 위치
c3 = pd.DataFrame(np.array(neigh_3_class),columns=['c'])
col3 = c3['c']
plt.scatter(x=new_data[0,0], y=new_data[0,1], 
            color=setcolor(col3))


# In[81]:


df['species'].shape


# In[118]:


c = pd.DataFrame(np.array(neigh_3_class),columns=['c'])
col = c['c']


# In[110]:


df['species'].values


# In[119]:


col.shape


# In[120]:


col.values


# In[ ]:





# In[123]:


column_train = ['petal length (cm)', 'petal width (cm)']
neigh_7 = KNeighborsClassifier(n_neighbors = 7, weights = 'distance')


# In[124]:


neigh_7_train = neigh_7.fit(df[column_train], df['species'])


# In[125]:


neigh_7_class = neigh_7_train.predict(new_data)


# In[126]:


print(neigh_7_class)


# In[127]:


# 학습 자료 출력
plt.scatter(x=df['petal length (cm)'], y=df['petal width (cm)'], 
           color=setcolor(df['species']))

# 새 자료의 위치
c7 = pd.DataFrame(np.array(neigh_7_class),columns=['c'])
col7 = c7['c']
plt.scatter(x=new_data[0,0], y=new_data[0,1], 
            color=setcolor(col7))


# In[129]:


# n_neighbor  모델의 accuracy 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# In[146]:


X_train, X_test, y_train, y_test = train_test_split(df[column_train], 
                                                    df['species'], test_size=0.33)


# In[147]:


# n_neighbor=3로 학습시키기 
neigh3 = KNeighborsClassifier(n_neighbors=3, weights='distance')
neigh3.fit(X_train, y_train)


# In[155]:


# n_neighbor=7 로 학습시키기
neigh7 = KNeighborsClassifier(n_neighbors=7, weights='distance')
neigh7.fit(X_train, y_train)


# In[156]:


print(neigh3.predict(X_test))
print(neigh7.predict(X_test))


# In[134]:


# 학습시킨 모델의 정확도 테스트
print('------------ 3개의 이웃 데이터 -----------')
print(classification_report(y_test, neigh3.predict(X_test)))


# In[152]:


# 학습시킨 모델의 정확도 테스트
print('------------ 7개의 이웃 데이터 -----------')
print(classification_report(y_test, neigh7.predict(X_test)))


# In[153]:


# 학습시킨 모델의 정확도 테스트
print('------------ 3개의 이웃 데이터 -----------')
print(confusion_matrix(y_test, neigh3.predict(X_test)))


# In[154]:


# 학습시킨 모델의 정확도 테스트
print('------------ 7개의 이웃 데이터 -----------')
print(confusion_matrix(y_test, neigh7.predict(X_test)))


# In[ ]:





# In[2]:


import numpy as np
from sklearn import svm


# In[3]:


X = np.array([ [0,0], [1,1] ])
y = [0, 1]


# In[4]:


LinearSVM = svm.LinearSVC()
LinearSVM.fit(X, y)


# In[5]:


LinearSVM.predict([[2,2]])


# In[6]:


print(LinearSVM.coef_[0])
print(LinearSVM.intercept_[0])


# In[15]:


import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")


# In[17]:


w = LinearSVM.coef_[0]
print(w)
b = LinearSVM.intercept_[0]
slope = -w[0] / w[1]
xx = np.linspace(0, 1.5)
yy = slope * xx - b/w[1]
h0 = plt.plot(xx, yy, 'k-', label='Hyperplane')
plt.scatter(X[:, 0], X[:,1], c=y)
plt.legend()
plt.show()


# In[9]:





# In[ ]:





# In[ ]:





# In[10]:


import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt


# In[11]:


X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = [0, 1, 1, 0]


# In[12]:


SVM_XOR = svm.SVC()
SVM_XOR.fit(X, y)


# In[13]:


test_data = np.array([[0.8, 0.8], [0.2, 0.9]])
SVM_XOR.predict(test_data)





