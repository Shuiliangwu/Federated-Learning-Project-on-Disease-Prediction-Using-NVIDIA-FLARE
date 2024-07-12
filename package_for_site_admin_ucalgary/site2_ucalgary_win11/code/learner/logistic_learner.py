
import argparse
import csv
import datetime
import json
import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import os
import xgboost as xgb
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


# (1) import nvflare client API
from nvflare import client as flare

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the argument parser for command line arguments
def define_args_parser():
    parser = argparse.ArgumentParser(description="scikit learn logistic regression")
    parser.add_argument("--data_root_dir", type=str, help="root directory path to csv data file")
    parser.add_argument("--results_dir", type=str, help="directory path to save results")
    parser.add_argument("--model_config", type=str, help="Path to the model configuration JSON file")
    parser.add_argument("--no-probability", dest="probability", action='store_false', help="Flag to disable probability estimates")
    # parser.add_argument("--unique_id", type=str, help="Unique identifier for the session")
    parser.set_defaults(probability=True)
    parser.add_argument("--random_state", type=int, default=0, help="random state")
    parser.add_argument("--test_size", type=float, default=0.2, help="test ratio, default to 20%")
    parser.add_argument(
        "--skip_rows",
        type=str,
        default=None,
        help="""If skip_rows = N, the first N rows will be skipped, 
       if skiprows=[0, 1, 4], the rows will be skip by row indices such as row 0,1,4 will be skipped. """,
    )
    return parser

# Convert the data dictionary to a dictionary of dataset tuples
def to_dataset_tuple(data: dict):
    dataset_tuples = {}
    for dataset_name, dataset in data.items():
        dataset_tuples[dataset_name] = _to_data_tuple(dataset)
    return dataset_tuples

# Convert the data to a tuple of features, labels, and data_num
def _to_data_tuple(data):
    data_num = data.shape[0]
    # split to feature and label
    x = data.iloc[:, 1:]
    y = data.iloc[:, 0]
    return x.to_numpy(), y.to_numpy(), data_num

# Load the features from the header file
def load_features(feature_data_path: str) -> List:
    try:
        features = []
        with open(feature_data_path, "r") as file:
            # Create a CSV reader object
            csv_reader = csv.reader(file)
            line_list = next(csv_reader)
            features = line_list
        return features
    except Exception as e:
        raise Exception(f"Load header for path'{feature_data_path} failed! {e}")

# Load the data from the CSV file and split it into train and test sets
def load_data(
    data_path: str, data_features: List, random_state: int, test_size: float, skip_rows=None
) -> Dict[str, pd.DataFrame]:
    try:
        df: pd.DataFrame = pd.read_csv(
            data_path, names=data_features, sep=r"\s*,\s*", engine="python", na_values="?", skiprows=skip_rows
        )

        train, test = train_test_split(df, test_size=test_size, random_state=random_state)

        return {"train": train, "test": test}

    except Exception as e:
        raise Exception(f"Load data for path '{data_path}' failed! {e}")

# Standardize the features by removing the mean and scaling to unit variance
def transform_data(data: Dict[str, Tuple]) -> Dict[str, Tuple]:
    scaler = StandardScaler()
    scaled_datasets = {}
    for dataset_name, (x_data, y_data, data_num) in data.items():
        # Ensure x_data is numpy array
        if not isinstance(x_data, np.ndarray):
            x_data = np.array(x_data)
        x_scaled = scaler.fit_transform(x_data)
        scaled_datasets[dataset_name] = (x_scaled, y_data, data_num)
    return scaled_datasets

# Make predictions on the testing set and evaluate the model with flag to disable probability estimates
def evaluate_model(x_test, model, y_test, probability: bool):
    auc = None
    report = None
    if probability:  
        y_pred_proba = model.predict_proba(x_test)[:, 1]  # Get probabilities for the positive class
        y_pred = model.predict(x_test)
        # Evaluate the model
        auc = roc_auc_score(y_test, y_pred_proba)
        report = classification_report(y_test, y_pred)
    else:
        y_pred = model.predict(x_test)
        # Evaluate the model
        auc = roc_auc_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
    return auc, report

# Initialize the model based on the specified model type
def initialize_model(model_config, model_type):
    # Extract the parameters for the specified model type from the model configuration
    params = model_config['models'][model_type]
    
    if model_type == "logistic_regression":
        model = LogisticRegression(**params)
    # elif model_type == "lasso_logistic_regression":
    #     model = LogisticRegression(**params)
    # elif model_type == "elastic_net_logistic_regression":
    #     model = LogisticRegression(**params)
    # elif model_type == "svm_with_probability":
    #     model = SVC(**params)
    # elif model_type == "svm_without_probability":
    #     model = SVC(**params)
    # elif model_type == "ridge_classifier":
    #     model = RidgeClassifier(**params)
    # elif model_type == "SGDClassifier":
    #     model = SGDClassifier(**params)
    elif model_type == "Random_Forest":
        model = RandomForestClassifier(**params)
    elif model_type == "XGBoost":
        model = xgb.XGBClassifier(**params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model

def main():
    parser = define_args_parser()
    args = parser.parse_args()
    data_root_dir = args.data_root_dir
    results_dir = args.results_dir
    model_config = args.model_config
    probability = args.probability
    # current_time = args.unique_id
    random_state = args.random_state
    test_size = args.test_size
    skip_rows = args.skip_rows

    # Load the configuration & initialize the model
    if args.model_config:
        with open(args.model_config, 'r') as file:
            config = json.load(file)
            # Get the active model type
            model_type = config['active_model']
            # Initialize the model
            model = initialize_model(config, model_type)
            logger.info(f"=======>>>>>>   Initialized {model_type} with parameters: {config['models'][model_type]}")
    else:
        logger.error("=======>>>>>>    No configuration file specified. Exiting.")
        return

    # Create results directory if it doesn't exist
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # (2) initializes NVFlare client API
    flare.init()

    # Get the site name using the NVFlare client API & Load the features from the header file
    site_name = flare.get_site_name()
    feature_data_path = f"{data_root_dir}/{site_name}_header.csv"
    features = load_features(feature_data_path)

    # Construct the path to the data file for the current site
    data_path = f"{data_root_dir}/{site_name}.csv"
    
    # Load the data from the CSV file and split it into train and test sets
    data = load_data(
        data_path=data_path, data_features=features, random_state=random_state, test_size=test_size, skip_rows=skip_rows
    )

    # Convert the data dictionary to a dictionary of dataset tuples
    data = to_dataset_tuple(data)
    
    # Standardize the features by removing the mean and scaling to unit variance
    dataset = transform_data(data)
    
    # Get the training set data
    x_train, y_train, train_size = dataset["train"]
    # logger.info(f"==========>>>>>>>>>>    Training set size: {train_size} samples")
    
    # Get the testing set data
    x_test, y_test, test_size = dataset["test"]

    # Create the unique subfolder if it doesn't exist
    current_time = datetime.datetime.now().strftime("%m%d%Y_%H%M%S")
    model_type = config['active_model']
    subfolder_name = f"{model_type}_{current_time}"
    subfolder = os.path.join(results_dir, subfolder_name)
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)

    while flare.is_running():
        # (3) receives FLModel from NVFlare
        input_model = flare.receive()
        curr_round = input_model.current_round

        # Train the model on the training set
        model.fit(x_train, y_train)

        # (6) evaluate global model first.
        global_auc, global_report = evaluate_model(x_test, model, y_test, probability)
        logger.info(f"=======>>>>>>   current_round: {curr_round}, {site_name}: global model AUC: {global_auc:.4f}")
        # logger.info(f"=======>>>>>>   current_round: {curr_round}, {site_name}: global report: {global_report}")

        # Train the model on the training set
        model.fit(x_train, y_train)

        local_auc, local_report = evaluate_model(x_test, model, y_test, probability)
        logger.info(f"=======>>>>>>   current_round: {curr_round}, {site_name}: local model AUC: {local_auc:.4f}")
        # logger.info(f"=======>>>>>>   current_round: {curr_round}, {site_name}: local report: {local_report}")

        # Save results to CSV file
        results_file = os.path.join(subfolder, "results.csv")
        header = ["Round", "Site", "Global AUC", "Local AUC"]
        with open(results_file, "a") as file:
            writer = csv.writer(file)
            if file.tell() == 0:
                writer.writerow(header)
            writer.writerow([curr_round, site_name, round(global_auc, 4), round(local_auc, 4)])

        # Save reports to text files
        global_report_file = os.path.join(subfolder, f"global_report_{curr_round}_{site_name}.txt")
        with open(global_report_file, "w") as file:
            file.write(global_report)
        local_report_file = os.path.join(subfolder, f"local_report_{curr_round}_{site_name}.txt")
        with open(local_report_file, "w") as file:
            file.write(local_report)

        # (7) construct trained FL model
        if isinstance(model, (LogisticRegression, SGDClassifier, RidgeClassifier, SVC)):
            if hasattr(model, "coef_"):
                coef = np.array(model.coef_)
                params = {"coef": coef}
            if hasattr(model, "intercept_"):
                intercept = np.array(model.intercept_)
                params["intercept"] = intercept
        elif isinstance(model, (RandomForestClassifier, xgb.XGBClassifier)):
            feature_importances = np.array(model.feature_importances_)
            params = {"feature_importances": feature_importances}
        else:
            raise ValueError("Model type not supported for parameter extraction")
        # params = {"coef": model.coef_, "intercept": model.intercept_}
        metrics = {"accuracy": np.array(global_auc)}
        output_model = flare.FLModel(params=params, metrics=metrics)

        # (8) send model back to NVFlare
        flare.send(output_model)
   

if __name__ == "__main__":
    main()
    

