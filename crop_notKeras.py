import pandas as pd
import numpy as np
from tensorflow import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from keras.models import save_model
import pickle

dataset = pd.read_csv('datasets/crop_recommendation.csv')

df = pd.DataFrame(dataset)
df.columns = df.columns.str.lower()

X = df.drop('crop', axis=1)
y = df['crop']

# Convert the output strings into numeric equivalent
label_encoder = LabelEncoder()
y_label_encoded = label_encoder.fit_transform(y)

# Convert the y value to one-hot encoding
labels_categorical = to_categorical(y_label_encoded)

# Standardize the X
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, labels_categorical, test_size=0.2, random_state=42)

def build_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(X.shape[1],)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(labels_categorical.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = build_model()

history = model.fit(X_train, y_train, batch_size=8, epochs=100, validation_data=(X_test, y_test), shuffle=True, verbose=1,)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

y_pred_one_hot = model.predict(X_test)
y_pred = np.argmax(y_pred_one_hot, axis=1)
y_test_labels = np.argmax(y_test, axis=1)
print(label_encoder.inverse_transform(y_test_labels))

accuracy = accuracy_score(y_pred, y_test_labels)


print(f'Model accuracy: {accuracy*100:.2f}%')

# Save the model to HDF5 file
model.save('model_crop_recommendation.h5')

# Save the label encoder and scaler to pickle files
with open('label_encoder.pickle', 'wb') as le_file:
    pickle.dump(label_encoder, le_file)

with open('scaler.pickle', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Model and encoders saved successfully!")

