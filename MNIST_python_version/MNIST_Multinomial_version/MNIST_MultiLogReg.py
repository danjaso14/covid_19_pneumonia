# -*- coding: utf-8 -*-
"""
Created on Wednesday Mar 25 19:52:50 2020

@author: Daniel Jaso
"""
#%%
import numpy as np
import tensorflow
from tensorflow.keras.datasets import mnist
import time


#%%


## load mnist data set 

(X_train,Y_train), (X_test, Y_test) = mnist.load_data()   ## imported from tensorflow.keras library


#%%

X_train = X_train.reshape((60000, 28 * 28))     ## flatten images 
X_train = X_train.astype('float32')/255         ## scale data

X_test_orig = X_test.copy()
X_test = X_test.reshape((10000, 28 * 28))     ## flatten images 
X_test = X_test.astype('float32')/255         ## scale data


#%%
# The to_categorical function performs one-hot-encoding.
Y_train = tensorflow.keras.utils.to_categorical(Y_train)
Y_test = tensorflow.keras.utils.to_categorical(Y_test)


#%%
## Multiclass Logistic Regression using Stochastic Gradient Descent




## objective function
def objective_func(beta, X, Y):
    """ 
    Definition:     objective_func trains the multiclass logistic regression 
        Params:     beta is the weight that paramatrized the linear functions in order to map X to Y.
                    X is the features used to train the softmax function
                    Y is the classes used to train the softmax function
        Return:     returns the cross entropy loss function 
    
    """
    numRows, numFeatures = X.shape
    cost = 0.0
    
    for i in range(numRows): ## softmax acitvation function over rows
        xi = X[i,:]
        yi =  Y[i]
        dotProds = xi@beta
        terms = np.exp(dotProds)
        probs = terms / np.sum(terms)
        k = np.argmax(yi)
        cost += np.log(probs[k])
      
       
    return -cost


## Stochastic Gradient Descent
def gradient_desc(beta,X,Y):
    """ 
    Definition:     gradient_desc performs stochastic gradicent descent in order to funt the best local minima.
        Params:     beta is the weight that paramatrizes the linear functions that map X to Y.
                    X is the features used to train the softmax function
                    Y is the classes used to train the softmax function
        Return:     returns the the corresponding weights
    
    """
    numFeatures = len(X)                       ## numver of features
    numClass = len(Y)                          ## number of classes
    grad = np.zeros((numFeatures, numClass))   ## number of betas shapes as :  (features, classes)
    
    for k in range(numClass): ## softmax acitvation function over columns
        dotProds = X@beta
        terms = np.exp(dotProds)
        probs = terms / np.sum(terms)
        grad[:,k] = (probs[k] - Y[k])*X
                            
    return grad

    
## Multilclass Logistic Regression using Stochastic Gradient Descent  function
def  multiLogReg(X,Y,lr, epochs): 
    """ 
    Definition:     multiLogReg performs stochastic gradient descent  
        Params:     X is the features used to train the softmax function
                    Y is the classes used to train the softmax function
                    lr is the learning rate that represents the amount the weights will update on the training data
                    epochs is how many times forward and backward propagation will be done on the trainind data
        Return:     returns the corresponding beta weights and the cross entropy loss. 
    
    """        
    
    numSamples, numFeatures = X.shape
    allOnes = np.ones((numSamples,1))                ## creating bias term 
    X = np.concatenate((X,allOnes),axis=1)           ## adding bias term to the original X
    numFeatures = numFeatures+1
    
    numClass = Y.shape[1]
    beta = np.zeros((numFeatures, numClass))        ## initializing beta weights to zeros with same dim as features and classes
    cost = np.zeros(epochs)                         ## initialize an array in order to store the cross entropy loss per epoch
    

    
    for ep in range(epochs):
        
        cost[ep] = objective_func(beta, X, Y)       ## computes each cost per epoch
        
        
        for i in np.random.permutation(numSamples): ## randomly iterates over all rows in order to eliminate biases
            
            beta = beta - lr*gradient_desc(beta, X[i],Y[i])  ## updates the beta weights 
            
        print("Epochs: " + str(ep+1) + " Cost: " + str(cost[ep]))
       
    return beta, cost


#%%

def multinomial_results(x_train, y_train, learning_rate, epochs, x_test, y_test):
    """ 
    Definition:     multinomial_results trains the neural network and tests it on the test data (unseen data).
        Params:     x_train is the features from the training set used to train the softmax function
                    y_train is the classes from the training set used to train the softmax function
                    learing_rate is the learning rate that represents the amount the weights will update on the training data
                    epochs is how many times forward and backward propagation will be done on the training data
                    x_test is the features from the test set.
                    y_test is the classes from the test.
        Return:     returns the time it took to compute each epoch as well as the percentage of correct classifications. 
    
    """  
    
    
    start = time.time()
    beta, cost = multiLogReg(x_train, y_train, learning_rate, epochs)
    end = time.time()
    timer = (end - start) ### took 3 minutes in my computer
    print("Time to compute Stochastic Gradient Descent: " + str(round(timer / 60, 2)) + " minutes " +" for " + str(epochs) + " epochs")
    
    numSamples, numFeatures = x_test.shape
    allOnes = np.ones((numSamples,1))
    X = np.concatenate((x_test,allOnes),axis=1)     ## add bias column
    
    numCorrect = 0
    for i  in range(numSamples):                    ## performs softmax function over test data
        xi = X[i,:]
        yi = y_test[i]
        dotProds = xi@beta                         ## apply beta weights previously trained in the neural network
        terms = np.exp(dotProds)
        probs = terms / np.sum(terms)
        k = np.argmax(probs)                      ## return the index with the maximum probability
                                                  ## this index represents the highest probability for the 
                                                  ## class the classfier predicts its correct class
        
        if yi[k] == 1:                           ## recall that one-hot-encoding was applied in the beginning 
            numCorrect += 1                      ## therefore if yi of that index equals to 1. This means that
                                                 ## it was correctly classified otherwise misclassified
            
    results = (numCorrect / numSamples)*100
    print("Accuracy: " + str(round(results, 2)) + " %")
        
#%%

multinomial_results(x_train= X_train, y_train= Y_train, 
                    learning_rate= 0.001, epochs=5,
                    x_test=X_test, y_test=Y_test)  









