import os
import mlflow
import mlflow.deployments
from mlflow.tracking import MlflowClient
from datetime import datetime

# Generate unique names using the current timestamp
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
model_name = f'HousePricePrediction-{timestamp}'
endpoint_name = f'house-price-prediction-endpoint-{timestamp}'

# AWS configuration
aws_region = os.getenv("AWS_REGION", "us-east-1")  # Change to your AWS region
execution_role_arn = os.getenv("EXECUTION_ROLE_ARN", "arn:aws:iam::207567773639:role/sagemakerops")  # Replace with your SageMaker execution role ARN
image_uri = "207567773639.dkr.ecr.us-east-1.amazonaws.com/mlflow-pyfunc:2.17.2"  # Replace with your image URI

# Set tracking URI from environment variable
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

# Get the latest model version in 'Production' stage from the Model Registry
client1 = MlflowClient()
versions = client1.get_latest_versions('HousePricePrediction01', stages=["Production"])
if not versions:
    raise ValueError(f"No version found for model 'HousePricePrediction01' in stage 'Production'")

latest_version = versions[0].version
model_uri = f"models:/HousePricePrediction01/{latest_version}"

# Deploy model to SageMaker
client = mlflow.deployments.get_deploy_client("sagemaker")

client.create_deployment(
    name=endpoint_name,
    model_uri=model_uri,
    config={
        "execution_role_arn": execution_role_arn,
        "region_name": aws_region,
        "instance_type": "ml.t2.medium",
        "instance_count": 1,
        "image_uri": image_uri,
        "mode": "replace"
    }
)

print(f"Model deployed to SageMaker and endpoint '{endpoint_name}' created.")
