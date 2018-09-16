# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 08:32:09 2018

@author: sudhakar
"""
from sklearn.metrics import classification_report
import numpy as np
import h5py
from utils import relu,sigmoid,cost,backward_relu,backward_sigmoid,preds
train_dataset = h5py.File('train_catvnoncat.h5', "r")
x_train = np.array(train_dataset["train_set_x"][:]) 
y_train = np.array(train_dataset["train_set_y"][:]) 

test_dataset = h5py.File('test_catvnoncat.h5', "r")
x_test = np.array(test_dataset["test_set_x"][:]) # your test set features
y_test= np.array(test_dataset["test_set_y"][:]) # your test set labels

classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
y_train = y_train.reshape((1, y_train.shape[0]))
y_test = y_test.reshape((1, y_test.shape[0]))
x_train =x_train.reshape(64*64*3,-1)
x_test = x_test.reshape(50,12288).T

x_train = x_train/255
x_test = x_test/255
np.random.seed(1)
w1 = np.random.randn(4,12288) -1
b1 = np.zeros((4,1))
w2 = np.random.randn(4,4) -1
b2 = np.zeros((4,1))
w3 = np.random.randn(4,4) - 1
b3 = np.zeros((4,1))
w4 = np.random.randn(1,4) - 1
b4 = np.zeros((1,1))
m= 209
def gradient(x,y,w1,w2,w3,w4,b1,b2,b3,b4,learning_rate,iterr):
    for i in range(iterr):
        z1 = np.dot(w1,x)+b1
        
            
        #print("w1 shape{fgh} x shape{yj}".format(fgh=w1.shape,yj = x.shape))
        a1 = relu(z1)
        
        z2 = np.dot(w2,a1)+b2
        a2 = relu(z2)
        z3 = np.dot(w3,a2)+b3
        a3 = relu(z3)
        z4 = np.dot(w4,a3)+b4
        a4 = sigmoid(z4)
        a4 = preds(a4)
        
        
        dz4 = a4 - y
        dw4 = 1/m*np.dot(dz4,a3.T)
        
        db4 = 1/m*np.sum(dz4)
        da3 = np.dot(w4.T,dz4)
        dz3 = da3*backward_relu(z3)
        dw3 = 1/m*np.dot(dz4,a2.T)
        db3 = 1/m*np.sum(dz3)
        da2 = np.dot(w3.T,dz3)
        dz2 = da2*backward_relu(z2)
        dw2 = 1/m*np.dot(dz2,a1.T)
        db2 = 1/m*np.sum(dz2)
        da1 = np.dot(w2.T,dz2)
        dz1 = da1 * backward_relu(z1)
        dw1 = 1/m*np.dot(dz1,x.T)
        db1 = 1/m*np.sum(dz1)
        #print(db1,dw4,db4,dw1)
        w1 = w1 - learning_rate*dw1
        b1 = b1 - learning_rate*db1
        w2 = w2 - learning_rate*dw2
        b2 = b2 - learning_rate*db2
        w3 = w3 - learning_rate*dw3
        b3 = b3 - learning_rate*db3
        w4 = w4 - learning_rate*dw4
        
        if (i% 100) == 0:
            error = a4-y
            print("Accuracy: "+str(np.sum((a4 == y)/m)))
            print("Error:" + str(np.mean(np.abs(error))))
            
    d = {'w1':w1,'b1':b1,'w2':w2,'b2':b2,'w3':w3,'b3':b3,'w4':w4,'b4':b4}
    
    return d
fg= gradient(x_train,y_train,w1,w2,w3,w4,b1,b2,b3,b4,0.01,1000)
def test(x,w1,w2,w3,w4,b1,b2,b3,b4):
    z1 = np.dot(w1,x)+b1
    a1 = relu(z1)
    z2 = np.dot(w2,a1)+b2
    a2 = relu(z2)
    z3 = np.dot(w3,a2)+b3
    a3 = relu(z3)
    z4 = np.dot(w4,a3)+b4
    a4 = sigmoid(z4)
    a4 = preds(a4)
    return a4
t = test(x_test,fg['w1'],fg['w2'],fg['w3'],fg['w4'],fg['b1'],fg['b2'],fg['b3'],fg['b4'])
print(classification_report(y_test,t))