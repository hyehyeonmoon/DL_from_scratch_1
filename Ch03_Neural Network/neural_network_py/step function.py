#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import matplotlib.pylab as plt


# In[2]:


#계단함수 구현하기
#넘파이 배열을 인수로 넣을 수 없음
def step_function(x):
    if x>0:
        return 1
    else:
        return 0


# In[3]:


#넘파이 배열을 인수로 넣을 수 있는 계단함수 구현
def step_function(x):
    y=x>0
    return y.astype(np.int)


# In[4]:


import numpy as np
x=np.array([-1,1,2])
y=x>0
y


# In[7]:


y=y.astype(np.int)
y


# In[9]:


# 최종
def step_function(x):
    return np.array(x > 0, dtype=np.int)

#그래프
X = np.arange(-5.0, 5.0, 0.1)
Y = step_function(X)
plt.plot(X, Y)
plt.ylim(-0.1, 1.1)  # y축의 범위 지정
plt.show()

