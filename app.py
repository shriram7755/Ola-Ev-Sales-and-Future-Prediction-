from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import base64
import pickle
import io

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    years = int(request.form['years'])

    # Load the data
    df = pd.read_csv('ev_sales.csv')

    # Load the trained SARIMA model
    model = pickle.load(open('sarimax_model.pkl', 'rb'))

    # Forecast sales
    forecast = model.forecast(steps=years * 12)

    # Convert the forecast to a list
    forecast_list = forecast.tolist()

    # Convert the plot to base64
    fig = plt.gcf()
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plot_data = base64.b64encode(buf.getbuffer()).decode('ascii')

    # Render the result page
    return render_template('result.html', plot_data=plot_data, forecast=forecast_list)

if __name__ == '__main__':
    app.run()
