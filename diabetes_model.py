import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import joblib
import os

# 1. Data Load
print("Loading diabetes data...")
df = pd.read_csv('../datasets/pima_diabetes.csv')
print(f"Dataset shape: {df.shape}")

# 2. Features aani Target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# 3. Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. SMOTE - Class imbalance fix
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# 5. Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. Model Train
print("Training Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
model.fit(X_train, y_train)

# 7. Accuracy Check
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Diabetes Model Accuracy: {accuracy*100:.2f}%")
print(classification_report(y_test, y_pred))

# 8. Model Save
os.makedirs('../models', exist_ok=True)
joblib.dump(model, '../models/diabetes_model.pkl')
joblib.dump(scaler, '../models/diabetes_scaler.pkl')
print("Model saved successfully!")