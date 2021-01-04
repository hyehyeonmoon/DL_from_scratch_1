#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pylab as plt


# In[3]:


#시그모이드 함수 구현
def sigmoid(x):
    return 1/(1+np.exp(-x))


# In[4]:


#확인
x=np.array([-1,1,2])
sigmoid(x)


# In[5]:


#그래프
X = np.arange(-5.0, 5.0, 0.1)
Y = sigmoid(X)
plt.plot(X, Y)
plt.ylim(-0.1, 1.1)
plt.show()

