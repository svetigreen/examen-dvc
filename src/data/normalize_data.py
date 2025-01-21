import pandas as pd
import os
import logging
from sklearn.preprocessing import StandardScaler
from pathlib import Path

def main(input_filepath, output_filepath):
    """ Normalizes training and test datasets
        Saves the scaled datasets to output_filepath.
    """
    logger = logging.getLogger(__name__)
    logger.info("Reading datasets for normalization.")

    # File paths
    X_train_file = os.path.join(input_filepath, "X_train.csv")
    X_test_file = os.path.join(input_filepath, "X_test.csv")
    X_train_scaled_file = os.path.join(output_filepath, "X_train_scaled.csv")
    X_test_scaled_file = os.path.join(output_filepath, "X_test_scaled.csv")

    # Import datasets
    X_train = import_dataset(X_train_file)
    X_test = import_dataset(X_test_file)

    # Normalize datasets
    X_train_scaled, X_test_scaled = normalize_datasets(X_train, X_test)

    # Create output folder if it doesn't exist
    create_folder_if_necessary(output_filepath)

    # Save the scaled datasets
    save_dataframe(X_train_scaled, X_train_scaled_file)
    save_dataframe(X_test_scaled, X_test_scaled_file)

    logger.info("Normalization complete. Scaled datasets saved.")

def import_dataset(file_path, **kwargs):
    return pd.read_csv(file_path, **kwargs)

def normalize_datasets(X_train, X_test):
    # Normalizes datasets using StandardScaler
    X_train = X_train.drop(columns=['date'])
    X_test = X_test.drop(columns=['date'])
    
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    return X_train_scaled, X_test_scaled

def create_folder_if_necessary(output_folderpath):
    # Create folder if necessary
    if not os.path.exists(output_folderpath):
        os.makedirs(output_folderpath)

def save_dataframe(df, file_path):
    # Save dataframes to their respective output file paths
    df.to_csv(file_path, index=False)

if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    input_filepath = "data/processed_data"
    output_filepath = "data/processed_data"

    main(input_filepath, output_filepath)
