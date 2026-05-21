# 🏥 Medilife — AI-Powered Multi-Disease Diagnosis System

An intelligent healthcare application that uses Machine Learning and AI to predict multiple diseases with high accuracy.

## ✨ Features

- 🩸 **Diabetes Prediction** — Random Forest model using PIMA dataset
- 🔬 **Breast Cancer Detection** — ML model using Wisconsin dataset
- ❤️ **Heart Disease Prediction** — Random Forest with 13 clinical features
- 🫘 **Kidney Disease Detection** — ML model with blood/urine parameters
- 🤖 **AI Chatbot** — Groq LLM powered medical assistant
- 📊 **Confidence Scoring** — Every prediction shows confidence %
- 💡 **AI Explanation** — LLM explains each prediction in simple language

## 🛠️ Technologies Used

- **Backend:** Python, Flask, REST API
- **ML Models:** Scikit-learn, Random Forest, SMOTE
- **AI/LLM:** Groq API, Llama 3.1
- **Frontend:** HTML, CSS, JavaScript
- **Database:** PostgreSQL, SQLAlchemy
- **Libraries:** Pandas, NumPy, Joblib

## 📊 Model Accuracies

| Disease | Model | Accuracy |
|---------|-------|----------|
| Diabetes | Random Forest | 82% |
| Breast Cancer | Random Forest | 96% |
| Heart Disease | Random Forest | 85% |
| Kidney Disease | Random Forest | 97% |

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/saivilasmogal/medilife-ai.git

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install flask numpy pandas scikit-learn imbalanced-learn joblib flask-cors groq

# 4. Run the app
python app.py

# 5. Open browser
http://127.0.0.1:5000
```

## 👨‍💻 Built By

**Sai Mogal** — Data Analyst & Data Engineer
- 📧 saimogal005@gmail.com
- 🔗 [LinkedIn](https://linkedin.com/in/sai-mogal)
- 🐙 [GitHub](https://github.com/saivilasmogal)

## ⚠️ Disclaimer

This is an AI-assisted tool for educational purposes only. Always consult a qualified medical professional for diagnosis and treatment.
