#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys, os
sys.path.append(os.pardir)
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax


# In[ ]:


# 입력층 뉴런 : 784개, 출력층 뉴런 : 10개
# 은닉층은 총 두개, 첫 번째 은닉층 : 50개의 뉴런, 두 번째 은닉층 : 100개의 뉴런


# In[2]:


def get_data():
    (x_train, t_train), (x_test, t_test)= load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    with open("sample_weight.pkl","rb") as f: #학습된 가중치 매개변수
        network=pickle.load(f)
    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3) #소프트맥스 사용

    return y


# In[8]:


x, t=get_data()
network=init_network()

accuracy_cnt=0
for i in range(len(x)):
    y=predict(network, x[i])
    p=np.argmax(y) #확률이 가장 높은 원소의 인덱스를 반환
    if p==t[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))

