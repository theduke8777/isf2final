import streamlit as st
import pickle
import numpy as np
from keras.models import load_model
import pandas as pd
from tensorflow import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from keras.models import load_model

st.markdown(
    """
        <style>
            .custom-title {
                text-align: center;
                color: #282a8c;
                font-weight: bold;
            }            

        <style>
    """
    , unsafe_allow_html=True)

# Load the saved model and encoders
model = load_model('model_crop_recommendation.h5')

with open('label_encoder.pickle', 'rb') as le_file:
    label_encoder = pickle.load(le_file)

with open('scaler.pickle', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with st.container(border=True):
    st.markdown('<h3 class="custom-title">Crop Recommendation using KerasClassifier</h3>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.write('<h5>Environmental Conditions</h5>', unsafe_allow_html=True)
        temperature = st.number_input('Input the temperature value (celsius)')
        humidity = st.number_input('Input the humidity value (%)')
        ph_value = st.number_input('Input the phValue')
        rainfall = st.number_input('Input the rainfall value (mm)')

    with col2:
        st.write('<h5>Soil Properties</h5>', unsafe_allow_html=True)
        nitrogen = st.number_input('Input the nitrogenn value (kg/ha)')
        phosphorus = st.number_input('Input the phosphorus value (kg/ha)')
        potassium = st. number_input('Input the potassium value (kg/ha)')

btn_analyze = st.button('Analyze', type='primary', use_container_width=True)

if btn_analyze:
    input_data = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph_value, rainfall]])
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    predicted_crop_index = np.argmax(prediction)
    predicted_crop = label_encoder.inverse_transform([predicted_crop_index])[0]
    st.success(f'Recommended Crop: {predicted_crop}')

    

   