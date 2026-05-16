import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load encoder and scaler
with open('one_hot_encod_geo.pkl', 'rb') as file:
    label_encoder_geo = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app
st.title('Customer Churn Prediction')

# User inputs
geography = st.selectbox(
    'Geography',
    label_encoder_geo.categories_[0]
)

gender = st.selectbox(
    'Gender',
    label_encoder_gender.classes_
)

age = st.slider('Age', 18, 92, 35)

credit_score = st.slider(
    'Credit Score',
    300,
    900,
    650
)

balance = st.slider(
    'Balance',
    0.0,
    250000.0,
    50000.0
)

estimated_salary = st.slider(
    'Estimated Salary',
    0.0,
    200000.0,
    50000.0
)

tenure = st.slider('Tenure', 0, 10, 5)

num_of_products = st.slider(
    'Number of Products',
    1,
    4,
    1
)

has_cr_card = st.selectbox(
    'Has Credit Card',
    [0, 1]
)

is_active_member = st.selectbox(
    'Is Active Member',
    [0, 1]
)

# Prepare input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode Geography
geo_encoded = label_encoder_geo.transform([[geography]]).toarray()

geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=label_encoder_geo.get_feature_names_out(['Geography'])
)

# Combine one-hot encoded columns with input data
input_data = pd.concat(
    [input_data.reset_index(drop=True), geo_encoded_df],
    axis=1
)

# Match column order with training data
input_data = input_data[scaler.feature_names_in_]

# Scale input data
input_data_scaled = scaler.transform(input_data)

# Predict churn
prediction = model.predict(input_data_scaled)

prediction_proba = prediction[0][0]

# Display prediction probability
st.write(f'Churn Probability: {prediction_proba:.2f}')

# Display result
if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')
