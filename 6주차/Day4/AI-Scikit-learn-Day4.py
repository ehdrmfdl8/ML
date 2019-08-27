#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
data = pd.read_csv('c:/MLdata/weather_nominal.csv', sep=',')


# In[2]:


data


# In[20]:





# In[3]:


 


# In[4]:


 

# In[5]:


multinomial_model


# In[6]:


print(data['outlook'])


# In[ ]:





# In[ ]:





# In[7]:


# mapping dictionary 만들기
outlook_dic = {'overcast':0, 'rainy':1, 'sunny':2}
temperature_dic = {'cool':0, 'hot':1, 'mild':2}
humidity_dic = {'high':0, 'normal':1}
windy_dic = {False:0, True: 1}


# In[8]:


# 딕셔너리를 활용해 데이터 매핑
data['outlook'] = data['outlook'].map(outlook_dic)
data['temperature'] = data['temperature'].map(temperature_dic)
data['humidity'] = data['humidity'].map(humidity_dic)
data['windy'] = data['windy'].map(windy_dic)


# In[9]:


data


# In[10]:


 


# In[11]:


#오늘의 날씨에 대한 예측
 


# In[12]:


# 계산된 확률
multinomial_model.predict_proba([[2, 2, 0, 1]])


# In[32]:


#오늘의 날씨에 대한 예측
 


# In[33]:


# 계산된 확률
multinomial_model.predict_proba([[1, 2, 0, 1]])


# In[15]:


from sklearn.datasets import load_iris
iris = load_iris()


# In[16]:


iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target


# In[17]:


iris_df.head()


# In[18]:


# 가우시안 모델 인스턴스화
 from sklearn.naive_bayes import GaussianNB
gaussian_model = GaussianNB()


# In[19]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris_df.iloc[:,:4], 
                                                    iris_df['species'],
                                                   test_size = 0.33)


# In[20]:


gaussian_model.fit(X_train, y_train)


# In[21]:


# 성능 평가
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, gaussian_model.predict(X_test)))


# In[22]:


print(confusion_matrix(y_test, gaussian_model.predict(X_test)))


# In[ ]:





# In[23]:


import pandas as pd
data = pd.read_csv('c:/MLdata/weather_nominal.csv', sep=',')


# In[24]:


# mapping dictionary 만들기
outlook_dic = {'overcast':0, 'rainy':1, 'sunny':2}
temperature_dic = {'cool':0, 'hot':1, 'mild':2}
humidity_dic = {'high':0, 'normal':1}
windy_dic = {False:0, True: 1}


# In[25]:


# 딕셔너리를 활용해 데이터 매핑
data['outlook'] = data['outlook'].map(outlook_dic)
data['temperature'] = data['temperature'].map(temperature_dic)
data['humidity'] = data['humidity'].map(humidity_dic)
data['windy'] = data['windy'].map(windy_dic)


# In[26]:


data


# In[27]:


data.info()


# In[28]:


# Decision Tree classifier instance 생성
 


# In[68]:





# In[29]:


# 결정 트리 분류기 학습
 


# In[30]:


 


# In[31]:

 

# In[34]:


# 계산된 확률
clf.predict_proba([[1, 2, 0, 1]])


# In[ ]:





# In[ ]:




