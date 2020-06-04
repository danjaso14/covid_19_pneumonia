import numpy as np
import matplotlib.pyplot as plt
import cv2 
import os 
import warnings
from tqdm import tqdm 
import random
import itertools as it
import time

import tensorflow
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers

from collections import Counter
from sklearn.model_selection import train_test_split
import numpy as np
from scipy import signal

warnings.filterwarnings('ignore')





def pre_processing(directory):
    image_size = 150

    data_log = []
    labels_log = []


    counter = 0
    print("Loading train images folder for patients diagnoses with Normal lungs")
    for normal_image in tqdm(os.listdir(directory + 'TrainData' + '/Normal/')): 
        if counter < 500:
            path = os.path.join(directory + 'TrainData' + '/Normal/', normal_image)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
            if img is None:
                continue
            img = cv2.resize(img, (image_size, image_size)).flatten()   
            np_img=np.asarray(img)
            data_log.append(img)
            labels_log.append(0)
            counter += 1
        else:
            break



    counter = 0
    print("Loading train images folder for patients diagnoses with Bacterial Pneumonia")
    for bact_image in tqdm(os.listdir(directory + 'TrainData' + '/BacterialPneumonia/')): 
        if counter < 500:
            path = os.path.join(directory + 'TrainData' + '/BacterialPneumonia/', bact_image)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
            if img is None:
                continue
            img = cv2.resize(img, (image_size, image_size)).flatten()   
            np_img=np.asarray(img)
            data_log.append(img)
            labels_log.append(1)
            counter +=1
        else:
            break


    counter = 0
    print("Loading train images folder for patients diagnoses with Viral Pneumonia")
    for viral_image in tqdm(os.listdir(directory + 'TrainData' + '/ViralPneumonia/')):
        if counter < 500:
            path = os.path.join(directory + 'TrainData' + '/ViralPneumonia/', viral_image)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
            if img is None:
                continue
            img = cv2.resize(img, (image_size, image_size)).flatten()   
            np_img=np.asarray(img)
            data_log.append(img)
            labels_log.append(2)
            counter += 1
        else:
            break

    counter = 0
    for viral_image in tqdm(os.listdir(directory + 'ValData' + '/ViralPneumonia/')): 
        if counter < 88:
            path = os.path.join(directory + 'ValData' + '/ViralPneumonia/', viral_image)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
            if img is None:
                continue
            img = cv2.resize(img, (image_size, image_size)).flatten()   
            np_img=np.asarray(img)
            data_log.append(img)
            labels_log.append(2)
            counter += 1
        else:
            break
        
        

    print("Loading train images folder for patients diagnoses with COVID-19 Pneumonia")
    for covid_image in tqdm(os.listdir(directory + 'TrainData' + '/COVID-19/')): 
        path = os.path.join(directory + 'TrainData' + '/COVID-19/', covid_image)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
        if img is None:
            continue
        img = cv2.resize(img, (image_size, image_size)).flatten()   
        np_img=np.asarray(img)
        data_log.append(img)
        labels_log.append(3)
        

    for covid_image in tqdm(os.listdir(directory + 'TrainData' + '/OversampledAugmentedCOVID-19/COVID-19')): 
        path = os.path.join(directory + 'TrainData' + '/OversampledAugmentedCOVID-19/COVID-19', covid_image)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
        if img is None:
            continue
        img = cv2.resize(img, (image_size, image_size)).flatten()   
        np_img=np.asarray(img)
        data_log.append(img)
        labels_log.append(3)
        
    for covid_image in tqdm(os.listdir(directory + 'ValData' + '/COVID-19/')): 
        path = os.path.join(directory + 'ValData' + '/COVID-19/', covid_image)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
        if img is None:
            continue
        img = cv2.resize(img, (image_size, image_size)).flatten()   
        np_img=np.asarray(img)
        data_log.append(img)
        labels_log.append(3)

    x_train, x_test, y_train, y_test = train_test_split(data_log, labels_log,
                                                    stratify=labels_log, 
                                                    test_size=0.3,
                                                    random_state=27)

    ## re shaping numpy array
    x_train = np.array(x_train)
    y_train = np.array(y_train).reshape(-1, 1)
    x_test = np.array(x_test)
    y_test = np.array(y_test).reshape(-1, 1)


    ## min max scaling
    x_train = (x_train-np.min(x_train))/(np.max(x_train)-np.min(x_train))
    x_test = (x_test-np.min(x_test))/(np.max(x_test)-np.min(x_test))


    return x_train, y_train, x_test, y_test

def oversample(X_train, Y_train, X_test, Y_test):
    """ 
    Definition:     oversample function takes the minr class (label) and increases it size with 
                    the biggest label in order to acheive a balance data set.
        Params:     x_train which are the features that we will train our model with
                    y_train which is the target variable that we will train our model with
                    x_test which are the feature will test how accurate our model is
                    y_test which is the target variabe that will test how accuracy our model is
        Return:     returns new_xtrain, new_ytrain, new_xtest, new_ytest
    
    """


    idx_train = np.where(Y_train == 3)[0]
    idx_test = np.where(Y_test == 3)[0]
    
    t1 = signal.resample(X_train[idx_train], len(X_train[idx_train])*4)[:350]
    t2 = signal.resample(X_test[idx_test], len(X_test[idx_test])*4)[:150]
    
    
    
    temp_train_idx = np.where(Y_train != 3)[0]
    temp_test_idx = np.where(Y_test != 3)[0]
    
    new_xtrain = np.concatenate((X_train[temp_train_idx],t1), axis=0)
    new_xtest = np.concatenate((X_test[temp_test_idx],t2), axis=0)
    
    
    s = np.repeat(3,350).reshape((350,1))
    new_ytrain = np.concatenate((Y_train[temp_train_idx],s), axis=0)
    s = np.repeat(3,150).reshape((150,1))
    new_ytest = np.concatenate((Y_test[temp_test_idx],s), axis=0)
    
    
    return new_xtrain, new_ytrain, new_xtest, new_ytest


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

def multinomial_results(X_train, Y_train, learning_rate, epochs, X_test, Y_test):
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
    
    y_train = tensorflow.keras.utils.to_categorical(Y_train)
    y_test = tensorflow.keras.utils.to_categorical(Y_test)
    
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
            
    accuracy = (numCorrect / numSamples)*100
    print("Accuracy: " + str(round(accuracy, 2)), "% " + 
          "using learning rate = " + str(learning_rate) + 
          " and epochs = " + str(epochs))

if __name__ == "__main__":

        x_train, y_train, x_test, y_test = pre_processing('/Users/student1/downloads/covid19-detection-xray-dataset (1)/')

        print("Results without oversampling")
        multinomial_results(X_train=x_train, Y_train=y_train, 
                       epochs=5, learning_rate=0.01,
                       X_test=x_test, Y_test=y_test)
        
        multinomial_results(X_train=x_train, Y_train=y_train, 
                       epochs=10, learning_rate=0.01,
                       X_test=x_test, Y_test=y_test)

        multinomial_results(X_train=x_train, Y_train=y_train, 
                       epochs=5, learning_rate=0.001,
                       X_test=x_test, Y_test=y_test)
        
        multinomial_results(X_train=x_train, Y_train=y_train, 
                       epochs=10, learning_rate=0.001,
                       X_test=x_test, Y_test=y_test)

        multinomial_results(X_train=x_train, Y_train=y_train, 
                       epochs=5, learning_rate=0.0001,
                       X_test=x_test, Y_test=y_test)
        
        multinomial_results(X_train=x_train, Y_train=y_train, 
                       epochs=10, learning_rate=0.0001,
                       X_test=x_test, Y_test=y_test)

        print()
        print("Results with oversampling")
        print()
        x_train, y_train, x_test, y_test = oversample(X_train= x_train, Y_train = y_train, X_test= x_test, Y_test = y_test)

        multinomial_results(X_train=x_train, Y_train=y_train, 
                       epochs=5, learning_rate=0.01,
                       X_test=x_test, Y_test=y_test)
        
        multinomial_results(X_train=x_train, Y_train=y_train, 
                       epochs=10, learning_rate=0.01,
                       X_test=x_test, Y_test=y_test)

        multinomial_results(X_train=x_train, Y_train=y_train, 
                       epochs=5, learning_rate=0.001,
                       X_test=x_test, Y_test=y_test)
        
        multinomial_results(X_train=x_train, Y_train=y_train, 
                       epochs=10, learning_rate=0.001,
                       X_test=x_test, Y_test=y_test)

        multinomial_results(X_train=x_train, Y_train=y_train, 
                       epochs=5, learning_rate=0.0001,
                       X_test=x_test, Y_test=y_test)

        multinomial_results(X_train=x_train, Y_train=y_train, 
                       epochs=10, learning_rate=0.0001,
                       X_test=x_test, Y_test=y_test)




        
        


