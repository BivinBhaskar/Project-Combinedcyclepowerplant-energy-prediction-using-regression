#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# Function to get user input features
def user_input_features():
    amb_temp = st.sidebar.number_input('Ambient Temperature (in C)')
    coolant_temp = st.sidebar.number_input('Coolant Temperature (in C)')
    coolant_flow_rate = st.sidebar.number_input('Coolant Flow Rate(in l/min)')
    coolant_pressure = st.sidebar.number_input('Coolant Pressure (in atm)')

    data = {
        'amb_temp': amb_temp,
        'coolant_temp': coolant_temp,
        'coolant_flow_rate': coolant_flow_rate,
        'coolant_pressure': coolant_pressure
    }

    features = pd.DataFrame(data, index=[0])
    return features

# Load the pre-trained linear regression model
model = RandomForestRegressor(n_estimators=100,max_features=0.7,max_samples=0.6)
data = pd.read_csv(r'C:\Users\bivin\OneDrive\Desktop\Regrerssion_energy_production_data.csv', delimiter=';')
X = data.drop('energy_production', axis=1)
y = data['energy_production']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model.fit(X_train_scaled, y_train)

# Streamlit App
st.title('Energy Production Prediction App')

# Sidebar for User Input
st.sidebar.header('User Input Parameters')

# Collect user input
user_input = user_input_features()

# Standardize user input using the same scaler fitted on the training data
user_input_scaled = scaler.transform(user_input.values.reshape(1, -1))

# Make predictions
prediction = model.predict(user_input_scaled)

# Display prediction
st.subheader('Energy Production Prediction:')
st.write(f'The predicted energy production is: {prediction[0]:.2f} MW')

# Display model performance on test data
y_pred_test = model.predict(X_test_scaled)
rmse = mean_squared_error(y_test, y_pred_test, squared=False)
r2 = r2_score(y_test, y_pred_test)
st.subheader('Model Performance on Test Data:')
st.write(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
st.write(f'R-squared (R2): {r2:.2f}')


# In[ ]:




