# -*- coding: utf-8 -*-
"""
Created on Friday April 3 13:00:25 2020

@author: Daniel Jaso
"""

import tensorflow
from tensorflow.keras.datasets import mnist
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers

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
# Specificy neural network architecture
nn = models.Sequential()
nn.add(layers.Dense(32,activation = 'relu', input_shape = (28 * 28,)))
nn.add(layers.Dense(10, activation = 'softmax'))

sgd = optimizers.SGD(lr=0.01)
nn.compile(optimizer = sgd ,loss = 'categorical_crossentropy',
                metrics = ['accuracy'])


#%% 

# Train  neural network
nn.fit(X_train,Y_train, epochs = 5, batch_size = 64) ##specify epochs and batch size

#%%

# Run neural network on test set
test_acc = nn.evaluate(X_test,Y_test)[1]
print("Accuracy: " + str(round(test_acc*100, 2)) + " %")
















