# Transfer Learning 
## Classification Problem - Emergency vs Non-emergency Vehicle Classification

## 1. Import neccessary libraries
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from skimage.transform import resize

from keras.utils import to_categorical
#used to preprocess data according to VGG16
from keras.applications.vgg16 import preprocess_input
#for instantiating the model and loading the weights and biases
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import Dense, InputLayer

## 2. Load the data
#reading the csv file containing data labels
data = pd.read_csv('data/emergency_classification.csv')

print(data.head())


#getting the labels for images
y = data.emergency_or_not.values
#converting label to categorical i.e instead of 0/1 labels we have 2 columns emergency and non-emergency ,
#with only one of them is true for every image
y = to_categorical(y)

## 3. Pre-Process Data
'''
Steps : 
1. Pre-process the data as per model's requirement
2. Prepare training and validation set

'''

