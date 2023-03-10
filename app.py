# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 12:46:42 2023

@author: HP
"""

import csv
from flask import Flask, request, render_template
import pickle
import pandas as pd
from sklearn.metrics import mean_squared_error


# Create an app object using Flask
app = Flask(__name__)

# Load the previous trained model 
model = pickle.load(open('models/xgb_model.pkl', 'rb'))

# Load the y_2021 csv file for calculating the mean squared error
# reader = csv.reader(open('utils/y_2021.csv', 'rb'))
y_2021 = pd.read_csv('utils/y_2021.csv')


# Define a functions to be triggered when '/' route is hit.
@app.route('/')
def home_page():
    return render_template('index.html')


# Define a function to be triggered when '/predict' route is hit
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    X_data = pd.read_csv(file)
    
    y_pred_xgb_2021 = model.predict(X_data)
    mse_xgb_2021 = mean_squared_error(y_2021, y_pred_xgb_2021)
    L = [i for i in y_pred_xgb_2021]

    output = f"XGBRegressor Mean Squared Error: {mse_xgb_2021}"
    prected_data = f"The data is:\n {L}"

    return render_template('index.html', prediction_text=prected_data, error=output)


if __name__ == '__main__':
    app.run(debug=True)