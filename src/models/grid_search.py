import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np
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

    # regressor
    regressor = RandomForestRegressor(random_state=42, n_jobs=-1)

    # parameters for GridSearchCV
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    # GridSearchCV
    grid_search = GridSearchCV(
        estimator=regressor,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=5,
        verbose=2,
        n_jobs=-1,
    )

    # Perform the grid search
    grid_search.fit(X_train, y_train)

    # Get the best parameters and score
    best_params = grid_search.best_params_
    best_score = -grid_search.best_score_  # Convert to positive MSE
    print(f"Best Parameters: {best_params}")
    print(f"Best Cross-Validated MSE: {best_score}")

    # Train the best model
    best_model = grid_search.best_estimator_
    best_model.fit(X_train, y_train)

    # Save the best model to the models directory
    model_filename = './models/best_params.pkl'
    os.makedirs(os.path.dirname(model_filename), exist_ok=True)
    joblib.dump(best_model, model_filename)
    print(f"Best model saved successfully to {model_filename}.")

if __name__ == "__main__":
    main()
