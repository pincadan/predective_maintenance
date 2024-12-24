import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load the sensor data from a CSV file
sensor_data = pd.read_csv('sensor_data.csv')

# Extract the features and target variable
X = sensor_data.drop('Maintenance_Required', axis=1)
y = sensor_data['Maintenance_Required']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest regression model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = rf_model.predict(X_test)

# Calculate the mean absolute error
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae:.2f}')

# Function to predict maintenance requirements
def predict_maintenance(sensor_values):
    maintenance_prob = rf_model.predict([sensor_values])[0]
    if maintenance_prob > 0.5:
        return 'Maintenance Required'
    else:
        return 'Maintenance Not Required'

# Example usage
sensor_values = [1.5, 2.2, 3.1, 4.6]  # Example sensor values
maintenance_status = predict_maintenance(sensor_values)
print(f'Predicted Maintenance Status: {maintenance_status}')