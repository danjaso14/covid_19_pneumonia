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


def neural_network(X_train, Y_train, num_epochs, learning_rate, X_test, Y_test):
    # Specificy neural network architecture
    
    Y_train = tensorflow.keras.utils.to_categorical(Y_train)
    Y_test = tensorflow.keras.utils.to_categorical(Y_test)
    
    
    nn = models.Sequential()
    nn.add(layers.Dense(32,activation = 'relu', input_shape = (150 * 150,)))
    nn.add(layers.Dense(4, activation = 'softmax'))
    
    sgd = optimizers.SGD(lr=learning_rate)
    nn.compile(optimizer = sgd ,loss = 'categorical_crossentropy',
                    metrics = ['accuracy'])
    
    
    # Train  neural network
    nn.fit(X_train,Y_train, epochs = num_epochs , batch_size = 64) 
    
    
    # Run neural network on test set
    accuracy = nn.evaluate(X_test,Y_test)[1]
    print("Accuracy: " + str(round(accuracy*100, 2)) + "% " + 
          "using learning rate = " + str(learning_rate) + 
          " and epochs = " + str(num_epochs))



if __name__ == "__main__":
    


    x_train, y_train, x_test, y_test = pre_processing('/Users/student1/downloads/covid19-detection-xray-dataset (1)/')
    
    print("Results without oversampling")
    neural_network(X_train=x_train, Y_train=y_train, 
                   num_epochs=5, learning_rate=0.01,
                   X_test=x_test, Y_test=y_test)
    
    neural_network(X_train=x_train, Y_train=y_train, 
                   num_epochs=10, learning_rate=0.01,
                   X_test=x_test, Y_test=y_test)

    
    neural_network(X_train=x_train, Y_train=y_train, 
                   num_epochs=5, learning_rate=0.001,
                   X_test=x_test, Y_test=y_test)
    
    neural_network(X_train=x_train, Y_train=y_train, 
                   num_epochs=10, learning_rate=0.001,
                   X_test=x_test, Y_test=y_test)

    neural_network(X_train=x_train, Y_train=y_train, 
                   num_epochs=5, learning_rate=0.0001,
                   X_test=x_test, Y_test=y_test)
    
    neural_network(X_train=x_train, Y_train=y_train, 
                   num_epochs=10, learning_rate=0.0001,
                   X_test=x_test, Y_test=y_test)
    print()
    print("Results with oversampling")
    print()
    
    x_train, y_train, x_test, y_test = oversample(X_train= x_train, Y_train = y_train, X_test= x_test, Y_test = y_test)
    
    neural_network(X_train=x_train, Y_train=y_train, 
                   num_epochs=5, learning_rate=0.01,
                   X_test=x_test, Y_test=y_test)
    
    

    neural_network(X_train=x_train, Y_train=y_train, 
                   num_epochs=10, learning_rate=0.01,
                   X_test=x_test, Y_test=y_test)

    
    
    neural_network(X_train=x_train, Y_train=y_train, 
                   num_epochs=5, learning_rate=0.001,
                   X_test=x_test, Y_test=y_test)
    
    

    neural_network(X_train=x_train, Y_train=y_train, 
                   num_epochs=10, learning_rate=0.001,
                   X_test=x_test, Y_test=y_test)

    neural_network(X_train=x_train, Y_train=y_train, 
                   num_epochs=5, learning_rate=0.0001,
                   X_test=x_test, Y_test=y_test)
    
    neural_network(X_train=x_train, Y_train=y_train, 
                   num_epochs=10, learning_rate=0.0001,
                   X_test=x_test, Y_test=y_test)

