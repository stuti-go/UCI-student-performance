# data_loader.py
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd

def load_data():
    # Fetch dataset
    student_performance = fetch_ucirepo(id=320)
    
    # Extract features (X) and target (y)
    X = student_performance.data.features
    y = student_performance.data.targets

    # Identify categorical columns
    categorical_columns = X.select_dtypes(include=["object"]).columns
    
    # Apply Label Encoding to categorical columns
    label_encoders = {}
    for col in categorical_columns:
        label_encoders[col] = LabelEncoder()
        X[col] = label_encoders[col].fit_transform(X[col])

    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test
