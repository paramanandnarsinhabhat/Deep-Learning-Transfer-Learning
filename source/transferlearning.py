### 1. Importing Libraries and Data Preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from keras.utils import to_categorical
from keras.applications.vgg16 import preprocess_input, VGG16
from keras.models import Sequential
from keras.layers import Dense, InputLayer, Flatten
from keras.optimizers import Adam  # Using Adam optimizer

### 2. Load and Preprocess the Data
data = pd.read_csv('data/emergency_classification.csv')
X = []
for img_name in data.image_names:
    img = plt.imread('data/images/' + img_name)
    img = resize(img, (224, 224), anti_aliasing=True, mode='reflect')  # Resize images
    X.append(img)
X = np.array(X)
X = preprocess_input(X)  # Preprocess the data
y = to_categorical(data.emergency_or_not.values)

### 3. Split the Dataset
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=42)

### 4. Load the Pretrained VGG16 Model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.summary()  # Optional: to view model structure

### 5. Fine-Tune the Model
# Extract features
X_train_features = base_model.predict(X_train)
X_valid_features = base_model.predict(X_valid)

# Flatten extracted features
X_train_features_flat = X_train_features.reshape(X_train_features.shape[0], -1)
X_valid_features_flat = X_valid_features.reshape(X_valid_features.shape[0], -1)

# Normalize features
max_val = X_train_features_flat.max()
X_train_features_flat /= max_val
X_valid_features_flat /= max_val

# Define and Compile Model
model = Sequential([
    InputLayer(input_shape=(X_train_features_flat.shape[1],)),
    Dense(1024, activation='relu'),
    Dense(2, activation='softmax')
])
model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the Model
model.fit(X_train_features_flat, y_train, epochs=100, validation_data=(X_valid_features_flat, y_valid))

# Evaluate the Model
predictions = model.predict(X_valid_features_flat)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_valid, axis=1)
accuracy = np.mean(predicted_classes == true_classes)
print(f'Validation Accuracy: {accuracy * 100:.2f}%')
