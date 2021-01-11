#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# coding: utf-8


class MulLayer: #곱하기 layer
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y): #순전파
        self.x = x
        self.y = y                
        out = x * y

        return out

    def backward(self, dout): #역전파 곱하기는 계산 노드 바꾸기 때문
        dx = dout * self.y  # x와 y를 바꾼다.
        dy = dout * self.x

        return dx, dy


class AddLayer: #더하기 layer
    def __init__(self):
        pass

    def forward(self, x, y): #순전파
        out = x + y

        return out

    def backward(self, dout): #역전파, 그대로 흘려보냄
        dx = dout * 1
        dy = dout * 1

        return dx, dy

