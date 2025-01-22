import pandas as pd
import numpy as np
from joblib import load
import json
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score


# Load data
X_test = pd.read_csv('data/processed_data/X_test_scaled.csv')
y_test = pd.read_csv('data/processed_data/y_test.csv')
y_test = np.ravel(y_test)

def main(repo_path):
    # load the trained model
    model = load(repo_path / "models/best_model.pkl")

    # generate predictions
    predictions = model.predict(X_test)

    # save predictions to a CSV file
    predictions_path = repo_path / "data/prediction.csv"
    predictions_df = pd.DataFrame({"y_test": y_test, "predictions": predictions})
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Predictions saved to {predictions_path}")


    # evaluation metrics
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    metrics = {"mse": mse, "r2": r2}

    # save metrics to a JSON file
    metrics_path = repo_path / "metrics/scores.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)  
    metrics_path.write_text(json.dumps(metrics, indent=4))
    print(f"Evaluation metrics saved to {metrics_path}")

if __name__ == "__main__":
    repo_path = Path(__file__).parent.parent.parent
    main(repo_path)
