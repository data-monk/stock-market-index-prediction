
# coding: utf-8

# <h1> Functions for modeling a Fully connected Deep Neural Network</h1>

# In[1]:

#import required library
import numpy as np


# In[2]:

# required activation functions
def sigmoid(z):
    return (1/(1+np.exp(-z)))

def relu(z):
    return np.maximum(0,z)

def tanh(z):
    return np.tanh(z)


# In[3]:

#required gradients of the activation functions
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


# In[4]:

#calculating activation 
def activation(z,func_name='sigmoid'):
    options={'relu':relu,
             'sigmoid':sigmoid,
             'tanh':tanh
             }
    return options[func_name](z)

#calculating gradients
def gradient(z,func_name="sigmoid"):
    options={'relu':relugrad,
             'sigmoid':sigmoidgrad,
             'tanh':tanhgrad
             }
    return options[func_name](z)


# In[5]:

#defining neural network architecture

def Neural_arch(data, label,num_of_layer,nn_each_layer):
    """ inputs:
              data: input data
              label:target label
              num_of_layer: number of layers in the neural network
              nn_each_layer: An array defining number of neurons in each layer 
        output:
               list of weights initialised randomly  for each layer
    """
    num_weight_mat=num_of_layer-1
    list_of_weight=[]
    for i in range(num_weight_mat):
        w=np.random.rand(nn_each_layer[i+1],nn_each_layer[i]+1)# since we are including bias here itself
        list_of_weight.append(w)
    return np.array(list_of_weight)


# In[6]:

#defining forward propagation
def forwardprop(X,list_of_weight):
    """  inputs: 
                X: input data
                list_of_weights: contains the list of weights for each layer
        outputs:
                list_of_z:pre-activation values of each neurons
                list_of_act:activation values of each neurons
                
    """
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


# In[7]:

def backprop(Y,list_of_z,list_of_act,list_of_weight):
    """ inputs:
                Y: target labels
                list_of_z: list of pre-activation for each neuron
                list_of_act: list of activation for each neuron
                list_of_weight: list of weights for each neuron
        output :
                list of gradients for each  weight or each neuron
                cost at that iteration
    """
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

