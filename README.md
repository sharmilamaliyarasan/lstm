# 📊 Web Traffic Forecasting Dashboard

## App Link:
https://6nuwftkuez5cc57c9rmrno.streamlit.app/

## 📌 Project Overview

This project is an interactive Streamlit dashboard that forecasts web page views using a trained LSTM/CNN model.

It provides users with:

Future traffic forecasts for customizable number of days

Peak traffic day detection

ROI prediction if ad spend data is available

Anomaly detection for recent data

CSV download for forecast results

The app covers the complete ML workflow:

Data preprocessing → scaling, encoding, and sequence generation

Model inference using a trained LSTM/CNN

Visualization → historical vs predicted traffic, ROI charts

## 🚀 Features

✅ Forecast web traffic using a trained deep learning model

✅ Adjustable forecast horizon via sidebar slider

✅ Optional ROI prediction based on conversion rate and ad spend

✅ Detect anomalies in recent traffic data

✅ Download predictions as CSV

✅ Modern dashboard with background image, animated title, and styled plots

## 🛠️ Tech Stack

Python 🐍

Streamlit → Interactive dashboard

TensorFlow/Keras → LSTM/CNN modeling

NumPy / Pandas → Data manipulation

Matplotlib → Plotting historical and predicted data

Joblib / h5py → Model/scaler persistence


## 📂 Project Structure

Web_Traffic_Forecast/

├── app.py

├── web_traffic_model.keras

├── scaler.pkl

├── final_website_stats.csv

├── images.jpeg

├── requirements.txt

└── README.md

## ⚙️ Installation & Setup

 Install dependencies:

pip install -r requirements.txt


Run the Streamlit app:

streamlit run app.py

## 📊 Example Workflow

Load historical web traffic data from CSV

Scale and preprocess features

Load trained LSTM/CNN model and scaler

Generate forecasts for selected number of days

Visualize historical vs predicted traffic

Detect anomalies in recent traffic

Optionally calculate ROI based on ad spend

Download results as CSV

## 📊 Evaluation Metrics

Prediction Accuracy → Depends on model performance and dataset

Forecast Plots → Visual comparison of historical and predicted traffic

ROI Curve → Predicted revenue vs ad spend

Anomaly Detection → Highlights traffic spikes or dips

## 🎯 Future Enhancements

Add multi-model support (CNN, MLP, Random Forest) in sidebar

Real-time traffic streaming input

Interactive plots with Plotly

Smooth section transitions/animations for modern UI

Deploy on Streamlit Cloud / Hugging Face for public access
