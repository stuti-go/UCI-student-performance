# compare_models.py
from linear_regression import train_linear_regression
from knn_model import train_knn
from random_forest import train_random_forest

def compare_models():
    # Train and evaluate each model
    print("Comparing models...\n")
    
    # Linear Regression
    mse_lr, r2_lr = train_linear_regression()
    
    # KNN Model
    mse_knn, r2_knn = train_knn()
    
    # Random Forest
    mse_rf, r2_rf = train_random_forest()
    
    # Print the comparison results
    print("\nComparison Results:")
    print(f"Linear Regression - MSE: {mse_lr:.2f}, R²: {r2_lr:.2f}")
    print(f"KNN - MSE: {mse_knn:.2f}, R²: {r2_knn:.2f}")
    print(f"Random Forest - MSE: {mse_rf:.2f}, R²: {r2_rf:.2f}")

if __name__ == "__main__":
    compare_models()
