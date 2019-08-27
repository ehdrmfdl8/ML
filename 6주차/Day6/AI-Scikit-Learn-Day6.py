#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[2]:


from skimage import data, color, feature
import skimage.data
import matplotlib.pyplot as plt
import numpy as np

image = color.rgb2gray(data.chelsea())
hog_vec, hog_vis = feature.hog(image, visualize=True)

fig, ax = plt.subplots(1, 2, figsize=(12, 6), 
                      subplot_kw = dict(xticks=[], yticks=[]))
ax[0].imshow(image, cmap='gray')
ax[0].set_title('input image')

ax[1].imshow(hog_vis)
ax[1].set_title('Visualization of HOG features')


# In[ ]:





# In[ ]:


# 안면 인식 알고리즘 구현: linear SVM 사용


# In[3]:


#1. 긍정 훈련 표본
from sklearn.datasets import fetch_lfw_people
faces = fetch_lfw_people()
positive_patches = faces.images
positive_patches.shape


# In[ ]:





# In[10]:


#2. 부정 훈련 표본
from skimage import data, transform
images_to_use = ['camera', 'text', 'coins', 'moon',
               'page', 'clock', 'immunohistochemistry', 
               'chelsea', 'coffee', 'hubble_deep_field']
images = [color.rgb2gray(getattr(data, name)())
          for name in images_to_use]
print(len(images))

fig, axes = plt.subplots(2,5, figsize=(9,4),
                         subplot_kw={'xticks':[], 'yticks':[]},
                        gridspec_kw=dict(hspace=0.1, wspace=0.1))                    
for i, ax in enumerate(axes.flat):
    ax.imshow(images[i], cmap='bone')


# In[11]:


from sklearn.feature_extraction.image import PatchExtractor
def extract_patches(img, N, scale=1.0, patch_size=positive_patches[0].shape):
    extracted_patch_size =     tuple((scale * np.array(patch_size)).astype(int))
    extractor = PatchExtractor(patch_size=extracted_patch_size,
                              max_patches=N, random_state=0)
    patches = extractor.transform(img[np.newaxis])
    if scale != 1:
        patches = np.array([transform.resize(patch, patch_size)
                           for patch in patches])
    return patches
negative_patches = np.vstack([extract_patches(im, 1000, scale)
                            for im in images for scale in [0.5, 1.0, 2.0]])
negative_patches.shape


# In[14]:


fig, ax = plt.subplots(6, 10)
for i, axi in enumerate(ax.flat):
    axi.imshow(negative_patches[500*i], cmap='gray')
    axi.axis('off')


# In[ ]:





# In[12]:


# 집합을 결합하고 HOG 특징 추출
from itertools import chain
X_train = np.array([feature.hog(im)
                   for im in chain(positive_patches, negative_patches)])
y_train = np.zeros(X_train.shape[0])
y_train[:positive_patches.shape[0]] = 1


# In[13]:


X_train.shape


# In[22]:


# Naive Bayes 의 얼굴 분류 성능
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

cross_val_score(GaussianNB(), X_train, y_train)


# In[ ]:





# In[29]:


from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(LinearSVC(), {'C':[1.0, 2.0, 4.0, 8.0]})
grid.fit(X_train, y_train)
grid.best_score_


# In[24]:


grid.best_params_


# In[ ]:





# In[28]:


model = grid.best_estimator_
model.fit(X_train, y_train)


# In[ ]:





# In[30]:


test_image = skimage.data.astronaut()
test_image = skimage.color.rgb2gray(test_image)
test_image = skimage.transform.rescale(test_image, 0.5)
test_image = test_image[:160, 40:180]

plt.imshow(test_image, cmap='gray')
plt.axis('off')


# In[ ]:





# In[31]:


def sliding_window(img, patch_size=positive_patches[0].shape,
                  istep=2, jstep=2, scale=1.0):
    Ni, Nj = (int(scale * s) for s in patch_size)
    for i in range(0, img.shape[0] - Ni, istep):
        for j in range(0, img.shape[1] - Ni, jstep):
            patch = img[i:i+Ni, j:j+Nj]
            if scale != 1:
                patch = transformation.resize(patch, patch_size)
            yield(i, j), patch

indices, patches = zip(*sliding_window(test_image))
patches_hog = np.array([feature.hog(patch) for patch in patches])
patches_hog.shape


# In[32]:


labels = model.predict(patches_hog)
labels.sum()


# In[33]:


fig, ax = plt.subplots()
ax.imshow(test_image, cmap='gray')
ax.axis('off')
Ni, Nj = positive_patches[0].shape
indices = np.array(indices)
for i, j in indices[labels==1]:
    ax.add_patch(plt.Rectangle((j,i), Nj, Ni, edgecolor='red',
                              alpha=0.3, lw=2, facecolor='none'))


# In[ ]:





# In[ ]:





# In[14]:


from sklearn.datasets import load_digits


# In[48]:


X, y = load_digits(return_X_y=True)
print(len(X))
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y,
                                                   test_size = 0.33)


# In[49]:


from sklearn.linear_model import Perceptron
slp = Perceptron(tol=1e-3, random_state=0, max_iter=200)


# In[50]:


slp.fit(X_train, y_train)


# In[51]:


slp.score(X_train, y_train)


# In[52]:


slp.score(X_test, y_test)


# In[44]:


from sklearn.neural_network import MLPClassifier


# In[60]:


mlp = MLPClassifier(solver='sgd', activation='logistic',
                    hidden_layer_sizes=(30, 10), 
                    random_state=1,max_iter=2000)


# In[61]:


mlp.fit(X_train,y_train)


# In[62]:


mlp.score(X_train, y_train)


# In[63]:


mlp.score(X_test, y_test)


# In[ ]:





# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




