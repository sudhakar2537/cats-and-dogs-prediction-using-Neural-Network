# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 08:36:55 2018

@author: sudhakar
"""

import numpy as np

def relu(Z):
    A = np.maximum(0,Z)
    return A
def sigmoid(z):
    A = 1/(1+np.exp(-z))
    return A
def cost(y_hat,y,m):
    
    cost = -1/m * np.sum(y * np.log(y_hat) + (1-y) * (np.log(1-y_hat)))
    return cost
def backward_sigmoid(z):
    s = 1/(1+np.exp(-z))
    return s*(1-s)
def backward_relu(z):
    z[z <= 0] = 0
    return z
def preds(d):
    for i in range(0, d.shape[1]):
        if d[0,i] >= 0.5:
            d[0,i] = 1
        else:
            d[0,i] = 0
    
    
    return d