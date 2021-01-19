#!/usr/bin/env python
# coding: utf-8

# In[7]:


import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from simple_convnet import SimpleConvNet
from common.trainer import Trainer
from collections import OrderedDict
from common.layers import *
from common.gradient import numerical_gradient


# In[ ]:


max_epochs = 20
network = SimpleConvNet(input_dim=(1,28,28), 
                        conv_param = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=10, weight_init_std=0.01)
                        
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=max_epochs, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr': 0.001},
                  evaluate_sample_num_per_epoch=1000)
trainer.train()


# ## 세세하게 따져보기

# In[2]:


input_dim=(1,28,28)
conv_param = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1}
hidden_size=100
output_size=10
weight_init_std=0.01

filter_num = conv_param['filter_num']
filter_size = conv_param['filter_size']
filter_pad = conv_param['pad']
filter_stride = conv_param['stride']
input_size = input_dim[1]
conv_output_size = (input_size - filter_size + 2*filter_pad) / filter_stride + 1
pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2))


# In[4]:


params = {}
params['W1'] = weight_init_std *                             np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
params['b1'] = np.zeros(filter_num)
params['W2'] = weight_init_std *                             np.random.randn(pool_output_size, hidden_size)
params['b2'] = np.zeros(hidden_size)
params['W3'] = weight_init_std *                             np.random.randn(hidden_size, output_size)
params['b3'] = np.zeros(output_size)


# In[9]:


layers = OrderedDict()
layers['Conv1'] = Convolution(params['W1'], params['b1'],
                                           conv_param['stride'], conv_param['pad'])
layers['Relu1'] = Relu()
layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
layers['Affine1'] = Affine(params['W2'], params['b2'])
layers['Relu2'] = Relu()
layers['Affine2'] = Affine(params['W3'], params['b3'])


# In[10]:


(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)
x_train, t_train = x_train[:5000], t_train[:5000]
x_test, t_test = x_test[:1000], t_test[:1000]


# In[12]:


x_train.shape #(5000, 1, 28, 28)


# In[13]:


t_train.shape #(5000), 정수 형식으로  되어 있는 벡터 


# In[17]:


x=x_train
for layer in layers.values():
    x=layer.forward(x)
    print(x.shape)


# ### poolinng의 output은 (30,12,12)가 맞는데 왜 Affine의 결과는 (5000, 100)

# In[ ]:


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        
        self.x = None
        self.original_x_shape = None
        # 가중치와 편향 매개변수의 미분
        self.dW = None
        self.db = None

    def forward(self, x):
        # 텐서 대응
        self.original_x_shape = x.shape #어떤 형식이든 다 (N=sample의 개수, 특성의 개수) 2차원 형태로 만듦
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out


# In[ ]:


#pooling output size : (5000,30,12,12) -> Affine을 통해서 (5000, 30*12*12)
#이에 따라서 첫번째 Affine의 W2의 형태도 (pool_output_size, hidden_size) 였음

