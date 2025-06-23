from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib


app = Flask(__name__)

model = joblib.load('random_forest_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/details')
def details():
    return render_template('details.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Retrieve form values
        gender = request.form['gender']
        hemoglobin = float(request.form['hemoglobin'])
        mch = float(request.form['mch'])
        mchc = float(request.form['mchc'])
        mcv = float(request.form['mcv'])

        # Gender encoding
        gender_encoded = 0 if gender == 'male' else 1

        # Feature array
        features = np.array([[gender_encoded, hemoglobin, mch, mchc, mcv]])

        # Prediction and confidence
        prediction = model.predict(features)[0]
        confidence = round(np.max(model.predict_proba(features)) * 100, 2)
        prediction_result = 'positive' if prediction == 1 else 'negative'

        # Recommendations
        recommendations = [
            "Increase iron-rich foods" if prediction == 1 else "Maintain a balanced diet",
            "Consult a doctor if symptoms persist" if prediction == 1 else "Stay hydrated and active"
        ]

        # Risk factors
        risk_factors = []
        if prediction == 1:
            if hemoglobin < 12:
                risk_factors.append(f"Low Hemoglobin: {hemoglobin} g/dL")
            if mch < 27:
                risk_factors.append(f"Low MCH: {mch} pg")
            if mchc < 32:
                risk_factors.append(f"Low MCHC: {mchc} g/dL")
            if mcv < 80:
                risk_factors.append(f"Low MCV: {mcv} fL")

        # User input for reference
        user_data = {
            "gender": gender,
            "hemoglobin": hemoglobin,
            "mch": mch,
            "mchc": mchc,
            "mcv": mcv
        }

        return render_template(
            'predict.html',
            prediction_result=prediction_result,
            confidence=confidence,
            user_data=user_data,
            recommendations=recommendations,
            risk_factors=risk_factors,
            submitted=True
        )

    # GET request
    return render_template('predict.html', submitted=False)



if __name__ == '__main__':
    app.run(debug=True)
