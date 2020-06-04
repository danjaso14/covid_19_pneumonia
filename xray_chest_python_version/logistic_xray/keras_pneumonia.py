#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Friday Apr 10 16:36:11 2020

@author: Daniel Jaso
"""
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
    
    ## min max scaling
    x_train = (x_train-np.min(x_train))/(np.max(x_train)-np.min(x_train))
    x_test = (x_test-np.min(x_test))/(np.max(x_test)-np.min(x_test))
    

        
    return x_train, y_train, x_test, y_test

def under_sampling(X_train, Y_train):
    
    labels_0 = np.where(Y_train == 0)[0]
    labels_1 = np.where(Y_train == 1)[0]
    
    random_label1 = random.choices(labels_1, k=1341)
    
    X_train = X_train[list(it.chain(*zip(labels_0, random_label1)))]        
    Y_train = Y_train[list(it.chain(*zip(labels_0, random_label1)))]
    
    return X_train, Y_train
        

def neural_network(X_train, Y_train, num_epochs, learning_rate, X_test, Y_test):
    # Specificy neural network architecture
    
    Y_train = tensorflow.keras.utils.to_categorical(Y_train)
    Y_test = tensorflow.keras.utils.to_categorical(Y_test)
    
    
    nn = models.Sequential()
    nn.add(layers.Dense(32,activation = 'relu', input_shape = (150 * 150,)))
    nn.add(layers.Dense(2, activation = 'sigmoid'))
    
    sgd = optimizers.SGD(lr=learning_rate)
    nn.compile(optimizer = sgd ,loss = 'categorical_crossentropy',
                    metrics = ['accuracy'])
    
    
    # Train  neural network
    nn.fit(X_train,Y_train, epochs = num_epochs , batch_size = 64) 
    
    
    # Run neural network on test set
    test_acc = nn.evaluate(X_test,Y_test)[1]
    print("Accuracy: " + str(round(test_acc*100, 2)) + "% " + 
          "using learning rate = " + str(learning_rate) + 
          " and epochs = " + str(num_epochs))

#%%


if __name__ == "__main__":
    
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
    
    tensorflow.logging.set_verbosity(tensorflow.logging.ERROR)

    
    warnings.filterwarnings('ignore')

    x_train, y_train, x_test, y_test = pre_processing('/Users/student1/downloads/chest_xray/')
    print("Results without under-sampling")
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
    print()
    print("Results with under-sampling")
    print()
    
    x_train, y_train = under_sampling(X_train=x_train, Y_train=y_train)
    
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
    

    
    

    