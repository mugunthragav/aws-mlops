import mlflow
from mlflow.tracking import MlflowClient
import os

def fetch_model_uri():
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://ec2-100-24-6-128.compute-1.amazonaws.com:5000")  # Replace with your MLflow URI

    model_name = "HousePricePrediction01"  # Replace with your model's name
    client = MlflowClient()

    # Get the latest version of the model in 'Production' stage
    versions = client.get_latest_versions(model_name, stages=["Production"])

    if not versions:
        raise ValueError(f"No version found for model '{model_name}' in stage 'Production'")

    latest_version = versions[0].version
    model_uri = f"models:/{model_name}/{latest_version}"

    # Set the model URI as an environment variable
    os.environ["MODEL_URI"] = model_uri
    print(f"Model URI: {model_uri}")

if __name__ == "__main__":
    fetch_model_uri()
