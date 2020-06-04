# Covid-19_image_classifier

Build up of a neural network by hand and compare its results with tensorflow over MNIST & Chest X-Ray Images (Pneumonia) datasets. 

## To train the network on your machine, first install all necessary dependencies using:

pip install -r requirements.txt

## Run Python MNIST data set without Keras

python -W ignore MNIST_python_version/MNIST_Multinomial_version/MNIST_MultiLogReg.py <br />
Rscript MNIST_R_version/MNIST_Multinomial_version/MNIST_MultiLogReg.R <br />


## Run Python or R MNIST data set with Keras

python -W ignore MNIST_python_version/MNIST_keras/MNIST_keras.py <br />
Rscript MNIST_R_version/MNIST_Keras_version/MNIST_MultiLogReg.R <br />


## Run Python on COVID-19 Detection X-Ray Dataset data set with Keras
#### Original data set is found here https://www.kaggle.com/darshan1504/covid19-detection-xray-dataset where this data set contains X-rays of COVID-19, Bacterial, Viral Pneumonia patients and Normal patients

python -W ignore xray_chest_python_version/multiLog_xray/keras_pneumonia.py

## Run Python on COVID-19 Detection X-Ray Dataset data set without Keras

python -W ignore xray_chest_python_version/multiLog_xray/multiLog_pneumonia.py


## Run Python on Chest X-Ray Images (Pneumonia) data set with Keras.
#### Original data set is found here https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia/ where this data set contains X-rays of Pneumonia and Normal patients

python -W ignore xray_chest_python_version/logistic_xray/keras_pneumonia.py


## Run Python on Chest X-Ray Images (Pneumonia) data set without Keras. 

python -W ignore xray_chest_python_version/logistic_xray/logistic_pneumonia.py