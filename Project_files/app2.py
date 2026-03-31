from flask import Flask, render_template, request, jsonify, redirect, url_for
import joblib
import numpy as np
import pandas as pd
import json
from urllib.parse import urlencode

app = Flask(__name__)

# =========================
# LOAD MODEL & COLUMNS ONCE
# =========================
model = joblib.load("heart_model.pkl")
columns = joblib.load("columns.pkl")

# =========================
# ROUTES
# =========================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/story')
def stories():
    return render_template('story.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/prediction_page')
def prediction_page():
    return render_template('prediction.html')

# =========================
# PREDICTION API
# =========================
@app.route('/prediction', methods=['POST'])
def predict():
    try:
        data = request.json
        print("Incoming data:", data)

        input_dict = {
            "BMI":             float(data.get('BMI', 0)),
            "Smoking":         int(data.get('Smoking', 0)),
            "AlcoholDrinking": int(data.get('AlcoholDrinking', 0)),
            "Stroke":          int(data.get('Stroke', 0)),
            "Sex":             int(data.get('Sex', 0)),
            "AgeCategory":     int(data.get('AgeCategory', 0)),
            "Diabetic":        int(data.get('Diabetic', 0)),
            "PhysicalActivity":int(data.get('PhysicalActivity', 0)),
            "SleepTime":       float(data.get('SleepTime', 7)),
            "Asthma":          int(data.get('Asthma', 0)),
            "KidneyDisease":   int(data.get('KidneyDisease', 0)),
            "DiffWalking":     int(data.get('DiffWalking', 0)),
            "GenHealth":       int(data.get('GenHealth', 3)),
            "PhysicalHealth":  float(data.get('PhysicalHealth', 0)),
            "MentalHealth":    float(data.get('MentalHealth', 0)),
            "SkinCancer":      int(data.get('SkinCancer', 0)),
        }

        # Feature engineering — must match heart_model.py exactly
        input_dict["RiskScore"] = (
            input_dict["Smoking"] +
            input_dict["AlcoholDrinking"] +
            input_dict["Stroke"] +
            input_dict["Diabetic"] +
            (0 if input_dict["PhysicalActivity"] == 1 else 1) +
            input_dict["KidneyDisease"] +
            input_dict["DiffWalking"]
        )

        bmi = input_dict["BMI"]
        if bmi < 18.5:
            input_dict["BMI_Category"] = 0
        elif bmi < 25:
            input_dict["BMI_Category"] = 1
        elif bmi < 30:
            input_dict["BMI_Category"] = 2
        else:
            input_dict["BMI_Category"] = 3

        input_dict["SleepRisk"]   = 1 if input_dict["SleepTime"] < 6 else 0
        input_dict["HighAgeRisk"] = 1 if input_dict["AgeCategory"] >= 8 else 0

        input_df = pd.DataFrame([input_dict])
        input_df = input_df[columns]

        probability = model.predict_proba(input_df)[0][1]
        if probability >= 0.3:
            prediction = "High Risk"
        elif probability >= 0.15:
            prediction = "Medium Risk"
        else:
            prediction = "Low Risk"

        if probability >= 0.3:  # High risk zone
             risk_score = 60 + (probability - 0.3) * 100
        elif probability >= 0.15:  # Medium risk zone
            risk_score = 25 + (probability - 0.15) * 150
        else:  # Low risk zone
            risk_score = probability * 150 


        # Final rounding
        prob_rounded = round(min(risk_score, 95), 2)
        print(f"Probability: {probability:.4f} -> {prediction}")

        # Return JSON to prediction.html (it handles localStorage + redirect)
        return jsonify({
            "prediction":  prediction,
            "probability": prob_rounded
        })

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({
            "prediction":  "Error",
            "probability": 0,
            "message":     str(e)
        })

# =========================
# RESULT PAGE
# =========================
@app.route('/result')
def result():
    return render_template('result.html')

# =========================
# RUN APP
# =========================
if __name__ == '__main__':
    app.run(debug=True)