# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 10:03:08 2017

@author: lucky
"""

import numpy as np
def sigmoid(z):
    return (1/(1+np.exp(-z)))
def relu(z):
    return np.maximum(0,z)
def tanh(z):
    return np.tanh(z)
def activation(z,func_name='sigmoid'):
    options={'relu':relu,
             'sigmoid':sigmoid,
             'tanh':tanh
             }
    return options[func_name](z)

def ind2vec(z,num_labels=-1):
    if num_labels==-1: 
       num_labels=max(z)+1
    result=np.zeros(z.shape[0],num_labels)
    for i in range(z.shape[0]):
        result[i,z[i]]=1
    return result    
def sigmoidgrad(z):
    return np.multiply(sigmoid(z),1-sigmoid(z))
def relugrad(z):
    if z<=0:
        grad=0
    else:
        grad=1
    return grad
def tanhgrad(z):
    gard=np.sech(z)**2
    return grad
    
def gradient(z,func_name="sigmoid"):
    options={'relu':relugrad,
             'sigmoid':sigmoidgrad,
             'tanh':tanhgrad
             }
    return options[func_name](z)
    
    
    
def Neural_arch(data, label,num_of_layer,nn_each_layer):
    num_weight_mat=num_of_layer-1
    list_of_weight=[]
    for i in range(num_weight_mat):
        w=np.random.rand(nn_each_layer[i+1],nn_each_layer[i]+1)# since we are including bias here itself
        list_of_weight.append(w)
    return np.array(list_of_weight)
    
    
    
def forwardprop(X,list_of_weight):
    m=X.shape[0] #number of instances
    bias=np.ones((m,1))
    list_of_z=[X]
    a1=np.concatenate((bias,X), axis=1)
    list_of_act=[a1]
    num_of_weights=list_of_weight.shape[0]
    for i in range (num_of_weights):
        z=np.dot(list_of_act[i],list_of_weight[i].T)
        if i<num_of_weights-1:
           a=np.concatenate((bias,activation(z)), axis=1)
        else:   
           a=z
        list_of_z.append(z)
        list_of_act.append(a)
        

    return list_of_z,list_of_act    

def backprop(Y,list_of_z,list_of_act,list_of_weight):
    m=list_of_z[0].shape[0]
    num_of_grad=list_of_weight.shape[0]
    list_of_grad=[0]*num_of_grad
    
#   grad=list_of_act[-1]-ind2vec(Y)
    grad=np.array(list_of_act[-1])-Y
    cost=np.sum((grad**2),axis=0)/m
    #print("Cost:{0}".format(cost))
    list_of_grad[-1]=-grad
     
    for j in range(2,num_of_grad+1):
#        m=list_of_z[-j].shape[0] #number of instances
#        bias=np.ones((m,1))
#        temp=np.concatenate((bias,gradient(list_of_z[-j])),axis=1)
        temp=gradient(list_of_z[-j])
        temp1=np.dot(list_of_grad[-j+1],list_of_weight[-j+1][:,1:])
      
        list_of_grad[-j]=np.multiply(temp1,temp)
    
    for i in range (num_of_grad):
        list_of_grad[i]=-1*(1/m)*np.dot(list_of_grad[i].T,list_of_act[i])      
    return np.array(list_of_grad),cost
#%%
import pandas as pd
data=pd.read_excel(r"data_stock.xlsx")
data=pd.read_csv("sp500.csv")
data=data.as_matrix()
data=data[1:,0].reshape(data[1:,0].shape[0],1)

def time_series_reshape(data,num_elements):
    num_rows=data.shape[0]-num_elements+1
    new_data=np.zeros((num_rows,num_elements))
    for i in range(num_rows):
        new_data[i,:]=data[i:i+num_elements].T
    return new_data    
    
a=2
data=time_series_reshape(data,a)   
 

#%%
import matplotlib.pyplot as plt
max_val=np.max(data,axis=0).reshape(1,-1)
min_val=np.min(data,axis=0).reshape(1,-1)
#print(min_val.shape)
data=(data-min_val)/(max_val-min_val)
np.random.shuffle(data)
x=data[:,0:a-1]
#x=(x-min_val)/(max_val-min_val)
y=data[:,-1].reshape(-1,1)


x_cos=np.cos(2*np.pi*x)
x_sin=np.sin(2*np.pi*x)
x_2=x**2

x=np.concatenate((x,x_cos,x_sin,x_2),axis=1)
x_train=x[0:3200,:]
y_train=y[0:3200,:]
x_test=x[3201:,:]
y_test=y[3201:,:]


input_size=x.shape[1]
#%%
lr=0.33
#plt.plot(x,y)
cost_plt=[]
list_of_weight=Neural_arch(x_train,y_train,3,[input_size,7,1])


for i in range(50000):
    list_of_z,list_of_act =forwardprop(x_train,list_of_weight)
    list_of_grad,cost=backprop(y_train,list_of_z,list_of_act,list_of_weight)
    list_of_weight=list_of_weight-lr*list_of_grad      
    if i%1000==0:
        print("cost :{0}".format(cost))
print("cost :{0}".format(cost))

   

k,result=forwardprop(x_test,list_of_weight)

#%%
plt.figure(figsize=(25,10))
plt.plot(result[-1][0:50])
plt.plot(y_test[0:50])
#%%
#plt.plot(result[-1][0:1000]-y_test[0:1000])