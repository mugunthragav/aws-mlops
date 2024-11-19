import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import pandas as pd
import joblib
import boto3

# Initialize S3 client
s3 = boto3.client('s3')
processed_bucket = 'data-bucket-house-processed-data'
model_bucket = 'model-bucket-house-model'

def train_and_log():
    # Download preprocessed data
    s3.download_file(processed_bucket, 'X_train.csv', 'X_train.csv')
    s3.download_file(processed_bucket, 'X_test.csv', 'X_test.csv')
    s3.download_file(processed_bucket, 'y_train.csv', 'y_train.csv')
    s3.download_file(processed_bucket, 'y_test.csv', 'y_test.csv')
    X_train = pd.read_csv('X_train.csv')
    X_test = pd.read_csv('X_test.csv')
    y_train = pd.read_csv('y_train.csv').squeeze()  # Convert to Series
    y_test = pd.read_csv('y_test.csv').squeeze()  # Convert to Series

    # Ensure correct number of features
    assert X_train.shape[1] == 12, "X_train should have 12 features"
    assert X_test.shape[1] == 12, "X_test should have 12 features"

    # Define models
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "LinearRegression": LinearRegression()
    }

    # Set MLflow experiment
    mlflow.set_tracking_uri("http://ec2-100-24-6-128.compute-1.amazonaws.com:5000")
    mlflow.set_experiment("HousePricePrediction01")

    best_model_name = None
    best_score = float('-inf')  # Initialize to lowest possible score

    # Train and evaluate models
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)

            # Log metrics and model
            mlflow.log_metric("test_score", score)
            mlflow.sklearn.log_model(model, "model")

            # Update best model
            if score > best_score:
                best_score = score
                best_model_name = name
                joblib.dump(model, 'best_model.pkl')
                s3.upload_file('best_model.pkl', model_bucket, 'best_model.pkl')
                print(f"New best model: {name} with score {score}")

    print("Training complete. Best model:", best_model_name)

if __name__ == "__main__":
    train_and_log()
