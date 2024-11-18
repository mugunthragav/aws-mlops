import mlflow
import mlflow.sagemaker as mfs
from mlflow.tracking import MlflowClient
import sys

def deploy_model():
    model_name = "HousePricePrediction01"  # Replace with your model name
    region = "us-east-1"                   # AWS region
    instance_type = "ml.t2.medium"         # Instance type for deployment
    instance_count = 1                     # Instance count
    execution_role_arn = "arn:aws:iam::207567773639:role/sagemakerops"  # Replace with your IAM role ARN

    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://ec2-100-24-6-128.compute-1.amazonaws.com:5000")  # Replace with your MLflow tracking URI

    # Get the latest model version in 'Production' stage
    client = MlflowClient()
    versions = client.get_latest_versions(model_name, stages=["Production"])
    if not versions:
        raise ValueError(f"No version found for model '{model_name}' in stage 'Production'")

    latest_version = versions[0].version
    model_uri = f"models:/{model_name}/{latest_version}"

    print(f"Deploying model from {model_uri} to SageMaker...")

    try:
        # Deploy the model to SageMaker using MLflow SageMaker API
        mfs.create_deployment(
            model_uri=model_uri,
            region_name=region,
            instance_type=instance_type,
            instance_count=instance_count,
            env={"DISABLE_NGINX": "true"}  # Optional, depending on your setup
        )
        print("Model deployed successfully!")
    except Exception as e:
        print(f"Error deploying model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    deploy_model()
