# preprocess.py
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Select only the required columns
    df_model = df[['Medu', 'Fedu', 'goout', 'Walc', 'failures', 'studytime', 'absences', 'G1', 'G2', 'G3']]

    # Separate features (X) and target variable (y)
    X = df_model.drop(columns=['G3'])
    y = df_model['G3']

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y
