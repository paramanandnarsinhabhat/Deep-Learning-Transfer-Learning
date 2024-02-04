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

# create an empty python list
X = []

# go through all the image locations one by one
for img_name in data.image_names:
    # read the image from location
    img = plt.imread('data/images/' + img_name)
    # pile it one over the other
    X.append(img)
    
# convert this python list to a single numpy array
X = np.array(X)

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
#show maximum and minimum values for the image array
X.min(), X.max()

print(X.min(), X.max())

#preprocess input images accordiing to requirements of VGG16 model
X = preprocess_input(X, data_format=None)

#print minimum and maximum values present in the array
X.min(), X.max()

print(X.min(), X.max()
)
# splitting the dataset into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=42)

## 4. Load weights of pretrained model
# creating model with pre trained imagenet weights
base_model = VGG16(weights='imagenet')

#shows model summary
base_model.summary()

# creating a VGG16 model with imagenet pretrained weights , accepting input of shape (224,224,3)
# also remove the final layers from model(include_top= False)
base_model = VGG16(weights='imagenet', input_shape=(224, 224, 3), include_top=False)

# show model summary
base_model.summary()

'''
## 5. Fine tune the model for the current problem
Steps:-
1. Extract features
2. Flatten the data
3. Rescale features
4. Create a Neural Network Model
5. Compile the model
6. Train and Validate the model
'''

# extract features using the pretrained VGG16 model
# for training set
base_model_pred = base_model.predict(X_train)
#for validation set
base_model_pred_valid = base_model.predict(X_valid)

#show shape of predictions
print(base_model_pred.shape)

# flattening the model output to one dimension for every sample of training set
base_model_pred = base_model_pred.reshape(1152, 7*7*512)

print(base_model_pred.shape)

print(base_model_pred_valid.shape)

# flattening the model output to one dimension for every sample of validation set
base_model_pred_valid = base_model_pred_valid.reshape(494, 7*7*512)

print(base_model_pred_valid.shape)

# checking the min and max of the extracted features
base_model_pred.min(), base_model_pred.max()

#get maximum value from generated features
max_val = base_model_pred.max()

#normalizing features generated from the VGG16 model to [0,1]
base_model_pred = base_model_pred / max_val
base_model_pred_valid = base_model_pred_valid / max_val
base_model_pred.min(), base_model_pred.max()

#create a sequential model 
model = Sequential()
# add input layer to the model that accepts input of shape 7*7*512
model.add(InputLayer((7*7*512, )))
# add fully connected layer with 1024 neurons and relu activation
model.add(Dense(units=1024, activation='relu'))
# add fully connected layer with 2 neurons and relu activation
model.add(Dense(units=2, activation='softmax'))

# compile the model
model.compile(optimizer='sgd', metrics=['accuracy'], loss='categorical_crossentropy')

model.summary()

# train model using features generated from VGG16 model
model.fit(base_model_pred, y_train, epochs=100, validation_data=(base_model_pred_valid, y_valid))

## 6. Get Predictions
# get predictions
predictions = model.predict_classes(base_model_pred_valid)
#show predictions
print(predictions)






