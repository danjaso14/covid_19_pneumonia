#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 16:58:42 2020

@author: student1
"""


import numpy as np
import cv2 
import os 
import warnings
from tqdm import tqdm 
import random
import itertools as it

import tensorflow 
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
import time
from scipy.special import expit, xlog1py



#%%
def pre_processing(directory):
    image_size = 150

    train_data_log = []
    train_labels_log = []
    
    print("Loading train images folder for patients diagnoses with normal lungs")
    for normal_image in tqdm(os.listdir(directory + 'train' + '/NORMAL/')): 
        path = os.path.join(directory + 'train' + '/NORMAL/', normal_image)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
        if img is None:
            continue
        img = cv2.resize(img, (image_size, image_size)).flatten()   
        train_data_log.append(img)
        train_labels_log.append(0)
        
    print("Loading train images folder for patients diagnoses with pneumonia lungs")
    for pneumonia_image in tqdm(os.listdir(directory + 'train' + '/PNEUMONIA/')): 
        path = os.path.join(directory + 'train' + '/PNEUMONIA/', pneumonia_image)
        img2 = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
        if img2 is None:
            continue
        img2 = cv2.resize(img2, (image_size, image_size)).flatten() 
        train_data_log.append(img2)
        train_labels_log.append(1)
    
    
    
    
    test_data_log = []
    test_labels_log = []
    
    print("Loading test images folder for patients diagnoses with normal lungs")
    for normal_image in tqdm(os.listdir(directory + 'test' + '/NORMAL/')): 
        path = os.path.join(directory + 'test' + '/NORMAL/', normal_image)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
        if img is None:
            continue
        img = cv2.resize(img, (image_size, image_size)).flatten()   
        test_data_log.append(img)
        test_labels_log.append(0)
    
    print("Loading test images folder for patients diagnoses with pneumonia lungs")
    for pneumonia_image in tqdm(os.listdir(directory + 'test' + '/PNEUMONIA/')): 
        path = os.path.join(directory + 'test' + '/PNEUMONIA/', pneumonia_image)
        img2 = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
        if img2 is None:
            continue
        img2 = cv2.resize(img2, (image_size, image_size)).flatten() 
        test_data_log.append(img2)
        test_labels_log.append(1)
    
    
    ## re shaping numpy array
    x_train = np.array(train_data_log)
    y_train = np.array(train_labels_log).reshape(-1, 1)
    x_test = np.array(test_data_log)
    y_test = np.array(test_labels_log).reshape(-1, 1)
    
    ## scaling
    x_train = (x_train-np.min(x_train))/(np.max(x_train)-np.min(x_train))
    x_test = (x_test-np.min(x_test))/(np.max(x_test)-np.min(x_test))
    
    
    return x_train, y_train, x_test, y_test

def undersampling(X_train, Y_train):
    
    labels_0 = np.where(Y_train == 0)[0]
    labels_1 = np.where(Y_train == 1)[0]
    
    random_label1 = random.choices(labels_1, k=1341)
    
    X_train = X_train[list(it.chain(*zip(labels_0, random_label1)))]        
    Y_train = Y_train[list(it.chain(*zip(labels_0, random_label1)))]
    
    return X_train, Y_train






#%%


## sigmoid function
def sigmoid(u):
    """ 
    Definition:     sigmoid performs sigmoid activation function 
        Params:     u is the argument of a dot prduct that is passed to 1 / (1 + np.exp(-u))
                    (e.g. Beta * X.T)
        Return:     returns sigmoid activation function computed
    
    """
    return 1 / (1 + np.exp(-u))

## objective function
def objective_func(beta, X, Y):
    """ 
    Definition:     objective_func trains the multiclass logistic regression 
        Params:     beta is the weight that paramatrized the linear functions in order to map X to Y.
                    X is the features used to train the softmax function
                    Y is the classes used to train the softmax function
        Return:     returns the cross entropy loss function 
    
    """
    numRows = X.shape[0]
    cost = 0.0
    
    for i in range(numRows):
        xi = X[i,:]
        yi =  Y[i]
        
#       p = sigmoid(np.dot(xi,beta))
#       cost += yi*(np.log(p)) + ((1 - yi)*(np.log(1-p)))
        
#       The expit function, also known as the logistic sigmoid function,
#       is defined as expit(x) = 1/(1+exp(-x)). It is the inverse of the logit function.
#       *** This avoids the issues of overflow that is display as Nan for the cost function
        
        p = expit(np.dot(xi,beta))                  
        cost += yi*(np.log(p)) + xlog1py(1-yi, p)  
        
        
    return -cost

       

## gradient evaluation function for Stochastic Gradient Descent
def gradient_desc(beta,X,Y):
    """ 
    Definition:     gradient_desc performs stochastic gradicent descent in order to funt the best local minima.
        Params:     beta is the weight that paramatrizes the linear functions that map X to Y.
                    X is the features used to train the softmax function
                    Y is the classes used to train the softmax function
        Return:     returns the the corresponding weights
    
    """
    p = sigmoid(np.dot(X,beta))
    grad = (p-Y)*X
    return grad
    

## Logistic Regression using Stochastic Gradient Descent  function
def  logReg(X,Y,t, epochs):         
    """ 
    Definition:     multiLogReg performs stochastic gradient descent  
        Params:     X is the features used to train the sigmoid function
                    Y is the classes used to train the sigmoid function
                    lr is the learning rate that represents the amount the weights will update on the training data
                    epochs is how many times forward and backward propagation will be done on the trainind data
        Return:     returns the corresponding beta weights and the cross entropy loss. 
    
    """ 
    
    numSamples, numCols = X.shape
    beta = np.random.randn(numCols)
    cost = np.zeros(epochs)
    
    for i in range(epochs):
        
        cost[i] = objective_func(beta, X, Y)
        
        for j in np.random.permutation(numSamples):
            
            beta = beta - t*gradient_desc(beta, X[j],Y[j])  
            
        print("Epochs: " + str(i + 1) + " Cost: " + str(cost[i]))

        
        
    return beta, cost




def logistic_results(X_train, Y_train, learning_rate, epochs, X_test, Y_test):
    """ 
    Definition:     logistic_results trains the neural network and tests it on the test data (unseen data).
        Params:     X_train is the features from the training set used to train the sigmoid function
                    Y_train is the classes from the training set used to train the sigmoid function
                    learing_rate is the learning rate that represents the amount the weights will update on the training data
                    epochs is how many times forward and backward propagation will be done on the training data
                    X_test is the features from the test set.
                    Y_test is the classes from the test.
        Return:     returns the time it took to compute each epoch as well as the percentage of correct classifications. 
    
    """  
    start = time.time()
    beta , cost = logReg(X_train, Y_train, learning_rate, epochs)
    end = time.time()
    timer = (end - start) 
    print("Time to compute Stochastic Gradient Descent: " + str(round(timer / 60, 2)) + " minutes " + " for " + str(epochs) + " epochs")
    
    numSamples, numFeatures = x_test.shape

    
    numCorrect = 0
    for i  in range(numSamples):                    ## performs sigmoid function over test data
        xi = x_test[i,:]
        yi = y_test[i]
        xi = x_test[i,:]
        yi =  y_test[i]
        p = sigmoid(np.dot(xi,beta)) 
        if p >= .5:
            p = 1
        else:
            p = 0
        
        if yi == p:
            numCorrect +=1
        
        
    
          
    results = (numCorrect / numSamples)*100
    print("Accuracy: " + str(round(results, 2)) + " %")
    





if __name__ == "__main__":
    


    x_train, y_train, x_test, y_test = pre_processing('/Users/student1/Downloads/chest_xray/')    
    print("Results without undersampling")
    
    logistic_results(X_train=x_train, Y_train= y_train, 
                   epochs=5, learning_rate=0.01,
                   X_test=x_test, Y_test=y_test)

    logistic_results(X_train=x_train, Y_train= y_train, 
                   epochs=10, learning_rate=0.01,
                   X_test=x_test, Y_test=y_test)

    logistic_results(X_train=x_train, Y_train= y_train, 
                   epochs=5, learning_rate=0.001,
                   X_test=x_test, Y_test=y_test)

    logistic_results(X_train=x_train, Y_train= y_train, 
                   epochs=10, learning_rate=0.001,
                   X_test=x_test, Y_test=y_test)



    print()
    print("Results with undersampling")
    print()
    
    x_train, y_train = undersampling(X_train= x_train, Y_train = y_train)

    logistic_results(X_train=x_train, Y_train= y_train, 
                   epochs=5, learning_rate=0.01,
                   X_test=x_test, Y_test=y_test)

    logistic_results(X_train=x_train, Y_train= y_train, 
                   epochs=10, learning_rate=0.01,
                   X_test=x_test, Y_test=y_test)

    logistic_results(X_train=x_train, Y_train= y_train, 
                   epochs=5, learning_rate=0.001,
                   X_test=x_test, Y_test=y_test)

    logistic_results(X_train=x_train, Y_train= y_train, 
                   epochs=10, learning_rate=0.001,
                   X_test=x_test, Y_test=y_test)









