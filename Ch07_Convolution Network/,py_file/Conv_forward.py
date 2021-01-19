#!/usr/bin/env python
# coding: utf-8

# In[4]:


import sys, os
sys.path.append(os.pardir)
from common.util import im2col
import numpy as np


# ## im2col 구현

# In[ ]:


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """다수의 이미지를 입력받아 2차원 배열로 변환한다(평탄화).
    
    Parameters
    ----------
    input_data : 4차원 배열 형태의 입력 데이터(이미지 수, 채널 수, 높이, 너비)
    filter_h : 필터의 높이
    filter_w : 필터의 너비
    stride : 스트라이드
    pad : 패딩
    
    Returns
    -------
    col : 2차원 배열
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant') #padding으로 높이와 너비에 padding을 넣어줌
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col


# ## im2col 시연

# In[34]:


x1=np.random.rand(1,3,3,3)
col1=im2col(x1, 2, 2, stride=1, pad=0)
print(col1.shape)


# In[35]:


print(x1)


# In[36]:


print(col1)


# In[39]:


x2=np.random.rand(1,3,7,7)
col2=im2col(x2, 5, 5, stride=1, pad=0)
print(col2.shape)


# In[40]:


x3=np.random.rand(10, 3, 7, 7)
col3=im2col(x3, 5, 5, stride=1, pad=0)
print(col3.shape)


# ## Simple Convolution

# In[ ]:


class Convolution:
    def __init__(self, W, b, strid=1, pad=0):
        self.W=W
        self.b=b
        self.stride=stride
        self.pad=pad
        
    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h=int(1+(H+2*self.pad-FH)/self.stride)
        out_w=int(1+(W+2*self.pad-FW)/self.stride)
        
        col=im2col(x, FH, FW, self.stride, self.pad)
        col_W=self.W.reshape(FN, -1).T
        out=np.dot(col, col_W)+self.b
        
        out=out.reshape(N, out_h, out_w, -1).transpose(0,3,1,2)
        
        return out
        


# self.W.reshape(FN, -1).T 확인해보기

# In[47]:


w=np.random.rand(2,3,2,2) #FN=2, N=23 H=2, W=2
print(w)


# In[49]:


w.reshape(2,-1) #FN=2


# In[53]:


w.reshape(2,-1).T


# In[52]:


col_w=w.reshape(2,-1).T


# np.dot(col, col_W)+self.b 형태 확인해보기

# In[55]:


out=np.dot(col1, col_w) #(pooling을 사용하지 않고, 온전한 합성곱 연산과 channel summation을 시행했을 때
out


# out.reshape(N, out_h, out_w, -1).transpose(0,3,1,2) 형태 확인해보기

# In[58]:


FN, C, FH, FW = w.shape
N, C, H, W = x1.shape
pad=0
stride=1
out_h=int(1+(H+2*pad-FH)/stride)
out_w=int(1+(W+2*pad-FW)/stride)


# In[61]:


print(FN, C, FH, FW, N, C, H, W, out_h, out_w)


# In[59]:


out.reshape(N, out_h, out_w, -1).transpose(0,3,1,2) # N, out_h, out_w, FN -> N, FN, out_h, out_w


# ### 데이터 개수가 2이상일 때(N>=2)

# In[62]:


x1=np.random.rand(2,3,3,3)
col1=im2col(x1, 2, 2, stride=1, pad=0)
print(col1.shape)


# In[67]:


w=np.random.rand(2,3,2,2) #FN=2, N=23 H=2, W=2
col_w=w.reshape(2,-1).T
print(col_w.shape)


# In[64]:


out=np.dot(col1, col_w) #(pooling을 사용하지 않고, 온전한 합성곱 연산과 channel summation을 시행했을 때
out


# In[68]:


FN, C, FH, FW = w.shape
N, C, H, W = x1.shape
pad=0
stride=1
out_h=int(1+(H+2*pad-FH)/stride)
out_w=int(1+(W+2*pad-FW)/stride)


# In[69]:


print(FN, C, FH, FW, N, C, H, W, out_h, out_w)


# In[74]:


result=out.reshape(N, out_h, out_w, -1).transpose(0,3,1,2) # N, out_h, out_w, FN -> N, FN, out_h, out_w
result


# In[73]:


print(result.shape)


# ## pooling 구현

# In[27]:


class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h=pool_h
        self.pool_w=pool_w
        self.stride=stride
        self.pad=pad
        
    def forward(self, x):
        N, C, H, W=x.shape
        out_h=int(1+(H-self.pool_h)/self.stride)
        out_w=int(1+(W-self.pool_w)/self.stride)
        
        #전개(1)
        col=im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col=col.reshape(-1, self.pool_h*self.pool_w)
        
        #최댓값(2)
        out=np.max(col, axis1)
        
        #성형(3)
        out=out.reshape(N, out_h, out_w, C).transpose(0,3,1,2)
        
        return out


# ### 예시(데이터가 1개일 때)

# In[82]:


x=np.random.rand(1,3,3,3)
col=im2col(x, 2, 2, stride=1, pad=0)
print(col.shape)


# In[83]:


print(col)


# In[88]:


w=np.random.rand(2,3,2,2) #FN=2, N=3 H=2, W=2
pool_h=2
pool_w=2
stride=1
pad=0
col2=col.reshape(-1, pool_h*pool_w)
print(col2.shape)


# In[85]:


print(col2)


# In[87]:


out=np.max(col2, axis=1)
print(out.shape)
print(out)


# In[90]:


N, C, H, W=x.shape
out_h=int(1+(H-pool_h)/stride)
out_w=int(1+(W-pool_w)/stride)

out.reshape(N, out_h, out_w, C).transpose(0,3,1,2)


# ### 예시(데이터가 2개 이상일 때)

# In[92]:


x=np.random.rand(2,3,3,3)
col=im2col(x, 2, 2, stride=1, pad=0)
print(col.shape)


# In[93]:


print(col)


# In[94]:


w=np.random.rand(2,3,2,2) #FN=2, N=3 H=2, W=2
pool_h=2
pool_w=2
stride=1
pad=0
col2=col.reshape(-1, pool_h*pool_w)
print(col2.shape)


# In[95]:


print(col2)


# In[96]:


out=np.max(col2, axis=1)
print(out.shape)
print(out)


# In[97]:


N, C, H, W=x.shape
out_h=int(1+(H-pool_h)/stride)
out_w=int(1+(W-pool_w)/stride)

out.reshape(N, out_h, out_w, C).transpose(0,3,1,2)

