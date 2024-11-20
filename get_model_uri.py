import mlflow
from mlflow.tracking import MlflowClient
import boto3

# Set tracking URI
mlflow.set_tracking_uri("http://ec2-100-24-6-128.compute-1.amazonaws.com:5000")

# Initialize S3 client
s3 = boto3.client('s3')
model_bucket = 'model-bucket-house-model'

def get_model_uri():
    # Load the best model name
    s3.download_file(model_bucket, 'best_model_name.txt', 'best_model_name.txt')
    with open('best_model_name.txt', 'r') as f:
        best_model_name = f.read().strip()

    # Initialize MLflow client
    client = MlflowClient()

    # Get the latest model version in 'Production' stage from the Model Registry
    versions = client.get_latest_versions(best_model_name, stages=["Production"])
    if not versions:
        raise ValueError(f"No version found for model '{best_model_name}' in stage 'Production'")

    latest_version = versions[0].version
    model_uri = f"models:/{best_model_name}/{latest_version}"

    return model_uri

if __name__ == "__main__":
    model_uri = get_model_uri()
    with open('model_uri.txt', 'w') as f:
        f.write(model_uri)
    print(f"Model URI saved to model_uri.txt: {model_uri}")
