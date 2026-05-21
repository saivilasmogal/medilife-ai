from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
import os
from groq import Groq

app = Flask(__name__)
CORS(app)

# Groq API Setup
client = Groq(api_key="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")

# Load all models
print("Loading all models...")

diabetes_model = joblib.load('models/diabetes_model.pkl')
diabetes_scaler = joblib.load('models/diabetes_scaler.pkl')

cancer_model = joblib.load('models/cancer_model.pkl')
cancer_scaler = joblib.load('models/cancer_scaler.pkl')

heart_model = joblib.load('models/heart_model.pkl')
heart_scaler = joblib.load('models/heart_scaler.pkl')

kidney_model = joblib.load('models/kidney_model.pkl')
kidney_scaler = joblib.load('models/kidney_scaler.pkl')

print("All models loaded successfully!")

# Groq AI Explanation
def get_ai_explanation(disease, prediction, confidence, features):
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful medical AI assistant. Always recommend consulting a doctor."
                },
                {
                    "role": "user",
                    "content": f"""A patient has been assessed for {disease}.
Prediction: {prediction}
Confidence: {confidence}%
Patient Data: {features}

In 3-4 simple sentences explain:
1. What this result means
2. Which values are concerning
3. What the patient should do next

Be empathetic and clear."""
                }
            ],
            max_tokens=200
        )
        return response.choices[0].message.content
    except Exception as e:
        return "Please consult a doctor for detailed analysis."

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Chatbot route
@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data['message']
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {
                    "role": "system",
                    "content": "You are Medilife AI assistant — a helpful medical chatbot. Always recommend consulting a doctor for serious concerns."
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ],
            max_tokens=200
        )
        reply = response.choices[0].message.content
        return jsonify({'reply': reply, 'status': 'success'})
    except Exception as e:
        return jsonify({'reply': 'Sorry, I am unable to answer right now.', 'status': 'error'})

# Diabetes prediction
@app.route('/predict/diabetes', methods=['POST'])
def predict_diabetes():
    try:
        data = request.get_json()
        features = np.array([[
            data['pregnancies'],
            data['glucose'],
            data['blood_pressure'],
            data['skin_thickness'],
            data['insulin'],
            data['bmi'],
            data['diabetes_pedigree'],
            data['age']
        ]])
        features_scaled = diabetes_scaler.transform(features)
        prediction = diabetes_model.predict(features_scaled)[0]
        confidence = diabetes_model.predict_proba(features_scaled)[0]
        confidence_score = round(max(confidence) * 100, 2)
        result = "Diabetes Detected" if prediction == 1 else "No Diabetes"
        explanation = get_ai_explanation(
            "Diabetes", result, confidence_score,
            f"Glucose:{data['glucose']}, BMI:{data['bmi']}, Age:{data['age']}"
        )
        return jsonify({'prediction': result, 'confidence': confidence_score, 'explanation': explanation, 'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'})

# Cancer prediction
@app.route('/predict/cancer', methods=['POST'])
def predict_cancer():
    try:
        data = request.get_json()
        features = np.array([list(data.values())])
        features_scaled = cancer_scaler.transform(features)
        prediction = cancer_model.predict(features_scaled)[0]
        confidence = cancer_model.predict_proba(features_scaled)[0]
        confidence_score = round(max(confidence) * 100, 2)
        result = "Malignant (Cancer)" if prediction == 0 else "Benign (No Cancer)"
        explanation = get_ai_explanation(
            "Breast Cancer", result, confidence_score,
            f"Radius:{list(data.values())[0]}, Texture:{list(data.values())[1]}"
        )
        return jsonify({'prediction': result, 'confidence': confidence_score, 'explanation': explanation, 'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'})

# Heart prediction
@app.route('/predict/heart', methods=['POST'])
def predict_heart():
    try:
        data = request.get_json()
        features = np.array([[
            data['age'], data['sex'], data['cp'],
            data['trestbps'], data['chol'], data['fbs'],
            data['restecg'], data['thalach'], data['exang'],
            data['oldpeak'], data['slope'], data['ca'], data['thal']
        ]])
        features_scaled = heart_scaler.transform(features)
        prediction = heart_model.predict(features_scaled)[0]
        confidence = heart_model.predict_proba(features_scaled)[0]
        confidence_score = round(max(confidence) * 100, 2)
        result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"
        explanation = get_ai_explanation(
            "Heart Disease", result, confidence_score,
            f"Age:{data['age']}, Chol:{data['chol']}, BP:{data['trestbps']}"
        )
        return jsonify({'prediction': result, 'confidence': confidence_score, 'explanation': explanation, 'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'})

# Kidney prediction
@app.route('/predict/kidney', methods=['POST'])
def predict_kidney():
    try:
        data = request.get_json()
        features = np.array([[
            data['age'], data['bp'], data['sg'],
            data['al'], data['su'], data['rbc'],
            data['pc'], data['pcc'], data['ba'],
            data['bgr'], data['bu'], data['sc'], data['hemo']
        ]])
        features_scaled = kidney_scaler.transform(features)
        prediction = kidney_model.predict(features_scaled)[0]
        confidence = kidney_model.predict_proba(features_scaled)[0]
        confidence_score = round(max(confidence) * 100, 2)
        result = "Kidney Disease Detected" if prediction == 1 else "No Kidney Disease"
        explanation = get_ai_explanation(
            "Kidney Disease", result, confidence_score,
            f"Age:{data['age']}, BP:{data['bp']}, Hemoglobin:{data['hemo']}"
        )
        return jsonify({'prediction': result, 'confidence': confidence_score, 'explanation': explanation, 'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)