import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from joblib import load, dump
from models.models_config import models  # Import the models dictionary
from mlflow.tracking import MlflowClient
import boto3
import os

# Initialize S3 client
s3 = boto3.client('s3')
processed_bucket = 'data-bucket-house-processed-data'
model_bucket = 'model-bucket-house-model'

mlflow.set_tracking_uri("http://your-mlflow-tracking-server:5000")
experiment_name = "House_Price_Prediction_Experiment"
mlflow.set_experiment(experiment_name)

# Initiate the MLflow client
client = MlflowClient()

def load_processed_data():
    # Download processed data from S3
    s3.download_file(processed_bucket, 'X_train.csv', 'X_train.csv')
    s3.download_file(processed_bucket, 'X_test.csv', 'X_test.csv')
    s3.download_file(processed_bucket, 'y_train.csv', 'y_train.csv')
    s3.download_file(processed_bucket, 'y_test.csv', 'y_test.csv')
    X_train = pd.read_csv('X_train.csv')
    X_test = pd.read_csv('X_test.csv')
    y_train = pd.read_csv('y_train.csv').squeeze()  # Convert to Series
    y_test = pd.read_csv('y_test.csv').squeeze()  # Convert to Series

    return X_train, X_test, y_train, y_test

def get_train_test_data(X_train, X_test, y_train, y_test, test_size, random_state):
    return train_test_split(X_train, X_test, test_size=test_size, random_state=random_state), train_test_split(y_train, y_test, test_size=test_size, random_state=random_state)

def train_all_models(X_train, X_test, y_train, y_test):
    best_model_name = None
    best_rmse = float('inf')

    for model_name, model_info in models.items():
        model_class = model_info["class"]
        parameters = model_info["parameters"]
        test_size = model_info["test_size"]
        random_state = model_info["random_state"]

        # Prepare the train/test data
        X_train_split, X_test_split, y_train_split, y_test_split = get_train_test_data(X_train, X_test, y_train, y_test, test_size, random_state)

        # Instantiate the model
        model_instance = model_class(**parameters)
        model_instance.fit(X_train_split, y_train_split)
        predictions = model_instance.predict(X_test_split)

        # Calculate RMSE
        rmse = mean_squared_error(y_test_split, predictions, squared=False)

        # Log model to MLflow
        with mlflow.start_run(nested=True):
            mlflow.sklearn.log_model(model_instance, model_name)
            mlflow.log_params(parameters)
            mlflow.log_metric("rmse", rmse)

            # Register the model
            mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/{model_name}", model_name)

            # Check if this model is the best
            if rmse < best_rmse:
                best_rmse = rmse
                best_model_name = model_name
                dump(model_instance, 'best_model.pkl')
                s3.upload_file('best_model.pkl', model_bucket, 'best_model.pkl')
                with open('best_model_name.txt', 'w') as f:
                    f.write(model_name)
                s3.upload_file('best_model_name.txt', model_bucket, 'best_model_name.txt')

        # End the current run
        mlflow.end_run()

    return best_model_name

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_processed_data()
    best_model_name = train_all_models(X_train, X_test, y_train, y_test)
    print(f"Best model: {best_model_name}")
