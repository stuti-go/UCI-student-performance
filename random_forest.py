# random_forest.py
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from preprocess import preprocess_data

def train_random_forest(X, y, n_estimators=100):
    # Split data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Random Forest model
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return mse, r2

if __name__ == "__main__":
    # Load preprocessed data
    file_path = 'student-mat.csv'
    X_scaled, y = preprocess_data(file_path)

    # Train Random Forest model and evaluate
    mse, r2 = train_random_forest(X_scaled, y)
    print(f"Random Forest Model - MSE: {mse}, R2: {r2}")
