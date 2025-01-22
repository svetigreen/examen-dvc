import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import yaml
import os

def main():
    # Load data
    X_train = pd.read_csv('data/processed_data/X_train_scaled.csv')
    X_test = pd.read_csv('data/processed_data/X_test_scaled.csv')
    y_train = pd.read_csv('data/processed_data/y_train.csv')
    y_test = pd.read_csv('data/processed_data/y_test.csv')

    # Flatten target variables
    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)

    # Load best parameters
    best_params = joblib.load("models/best_params.pkl")

    # Load training parameters from params.yaml
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    random_state = params["training"]["random_state"]
    n_jobs = params["training"]["n_jobs"]

    # Define the model
    model = RandomForestRegressor(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        min_samples_split=best_params["min_samples_split"],
        min_samples_leaf=best_params["min_samples_leaf"],
        random_state=random_state,
        n_jobs=n_jobs,
    )

    # Train the model
    print("Training the model...")
    model.fit(X_train, y_train)

    # Save the trained model
    model_filename = "./models/best_model.pkl"
    os.makedirs(os.path.dirname(model_filename), exist_ok=True)
    joblib.dump(model, model_filename)
    print(f"Model saved successfully to {model_filename}.")

if __name__ == "__main__":
    main()
