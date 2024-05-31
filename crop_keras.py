import pandas as pd
import numpy as np
from tensorflow import random
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import seaborn as sns
from scikeras.wrappers import KerasClassifier
import pickle


dataset = pd.read_csv('datasets/crop_recommendation.csv')

# Convert column names to lower caps
df = pd.DataFrame(dataset)
df.columns = df.columns.str.lower()

# get the features and output classes
X = df.iloc[:, :7]
y = df['crop']


# Convert the string into numeric value
label_encoder = LabelEncoder()
y_label_encoded = label_encoder.fit_transform(y)

# Standardize the X
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# build a model
def build_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(X.shape[1],)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(len(y.unique()), activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Wrap using KerasClassifier
classifier = KerasClassifier(build_fn=build_model, epochs=100, batch_size=10, verbose=0, shuffle=False)

# Define the cross validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross fold
results = cross_val_score(classifier, X_scaled, y_label_encoded, cv=kfold) 

classifier.fit(X_scaled, y_label_encoded)

with open('model_pickle_crop2.pickle', 'wb') as model_pickle:
    pickle.dump(classifier, model_pickle)

with open('output_pickle_crop2.pickle', 'wb') as output_pickle:
    pickle.dump(label_encoder, output_pickle)

with open('scaler_pickle_crop2.pickle', 'wb') as scaler_pickle:
    pickle.dump(scaler, scaler_pickle)

print("Model and encoders saved successfully!")

