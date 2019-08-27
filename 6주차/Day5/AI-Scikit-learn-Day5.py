#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np


# In[13]:


# 4개 영역의 2차원 자료의 생성
from sklearn.datasets.samples_generator import make_blobs
X, y_true = make_blobs(n_samples=300, centers=4, 
                       cluster_std = 0.60, random_state = 0)
plt.scatter(X[:, 0], X[:, 1], s=50)


# In[7]:


# k-means clustering



# In[8]:




# In[9]:


print(y_kmeans)


# In[10]:


# 그룹별로 색깔을 달리해 표현하기, 군집 중앙 표시
plt.scatter(X[:,0], X[:,1], c=y_kmeans, s=50, cmap = 'viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:,1], c='black', s=200, alpha=0.5)


# In[ ]:





# In[12]:


#비선형 경계를 가지는 자료의 경우 : KMeans는 직선으로 분할하기 때문에 제대로 자르지 못 한다.
from sklearn.datasets import make_moons
X, y = make_moons(200, noise=0.05, random_state = 0)
plt.scatter(X[:,0], X[:,1], c=y, s=50, cmap='viridis')


# In[56]:
labels = KMeans(2, random_state=0).fit_predict(X)
plt.scatter(X[:,0], X[:,1], c=labels, s=70, cmap='viridis', alpha=0.5 )



# In[71]:


# In[17]:


from sklearn.datasets import load_digits
digits = load_digits()
digits.data.shape


# In[18]:


# k-means clustering



# In[19]:


kmeans.cluster_centers_.shape


# In[24]:


# 64차원의 군집 10개
print(clusters.shape)
clusters


# In[13]:


# 군집 화면 출력


# In[28]: image를 표시하는 전형적인 방법
fig, ax = plt.subplots(2, 5, figsize=(8, 3))
centers = kmeans.cluster_centers_.reshape(10, 8, 8)
print(ax.shape)
print(ax)
print(centers.shape)
for axi, center in zip(ax.flat, centers):
    axi.set(xticks=[], yticks=[])
    axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)


# In[31]:


# 1과 8을 제외하면 군집으로도 숫자 인식 가능한 중심을 찾을 수 있음
print(clusters.shape)
clusters



# In[43]:


#PCA 를 위한 자료 준비
rng = np.random.RandomState(1)
X = np.dot(rng.rand(2,2), rng.randn(2, 200)).T
plt.scatter(X[:,0], X[:, 1])
plt.axis('equal')


# In[44]:

# In[45]:
from sklearn.decomposition import PCA
mypca = PCA(n_components=2)  #  n_components 값과 행렬의 행값은 같아야 한다.  왜?
mypca.fit(X)
print( '2---------- \n', mypca.components_ )

# In[46]:


print( mypca.explained_variance_ )


# In[67]:


def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops = dict(color='red',
                      arrowstyle='simple',
                     linewidth=2,
                     shrinkA=0, shrinkB=0 )
    ax.annotate('', v1, v0, arrowprops=arrowprops)

# data plotting
plt.scatter(X[:, 0], X[:,1], alpha=0.2)
for length, vector in zip(mypca.explained_variance_, mypca.components_):
    v = vector * 3* np.sqrt(length)
    draw_vector(mypca.mean_, mypca.mean_ + v)
plt.axis('equal')

# In[51]:


# 주성분을 하나로 만들기
dimpca = PCA(n_components=1)
dimpca.fit(X)
X_pca = dimpca.transform(X)
print('original shape: ', X.shape)
print('transformed shape: ', X_pca.shape)


# In[39]:


# 역변환 
X_new = dimpca.inverse_transform(X_pca)
plt.scatter(X[:,0], X[:,1], alpha = 0.2)
plt.scatter(X_new[:,0], X_new[:,1], alpha=0.8)
plt.axis('equal')

# In[76]:


# 고유 얼굴 성분 찾기 & 얼굴들 보기
from sklearn.datasets import fetch_lfw_people
faces = fetch_lfw_people(min_faces_per_person=60)
print(faces.target_names)
print(faces.images.shape)
fig, axes = plt.subplots(3,8, figsize=(9,4),
                         subplot_kw={'xticks':[], 'yticks':[]},
                        gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i, ax in enumerate(axes.flat):
    ax.imshow(faces.images[i].reshape(62,47), cmap='bone')


# In[73]:
#
face_pca = PCA(n_components=150)
face_pca.fit(faces.data)
print(face_pca.components_.shape)


# In[80]:

fig, axes = plt.subplots(3,8, figsize=(9,4),
                         subplot_kw={'xticks':[], 'yticks':[]},
                        gridspec_kw=dict(hspace=0.1, wspace=0.1))                    
for i, ax in enumerate(axes.flat):
    ax.imshow(face_pca.components_[i].reshape(62,47), cmap='bone')


