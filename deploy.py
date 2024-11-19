import mlflow
import mlflow.deployments
import boto3
from mlflow.tracking import MlflowClient
# AWS configuration
aws_region = "us-east-1"
execution_role_arn = "arn:aws:iam::207567773639:role/sagemakerops"  # Replace with your SageMaker execution role ARN
model_name = 'HousePricePrediction01'
model_version = '1'  # Replace with your model version if needed
endpoint_name = 'house-price-prediction-endpoint001'  # Replace with desired endpoint name
mlflow.set_tracking_uri(
    "http://ec2-100-24-6-128.compute-1.amazonaws.com:5000")  # Replace with your MLflow tracking URI

# Get the latest model version in 'Production' stage from the Model Registry
client1 = MlflowClient()
versions = client1.get_latest_versions(model_name, stages=["Production"])
if not versions:
    raise ValueError(f"No version found for model '{model_name}' in stage 'Production'")

latest_version = versions[0].version
model_uri = f"models:/{model_name}/{latest_version}"



# Deploy model to SageMaker
client = mlflow.deployments.get_deploy_client("sagemaker")

client.create_deployment(
    name=endpoint_name,
    model_uri=model_uri,
    config={
        "execution_role_arn": execution_role_arn,
        "region_name": aws_region,
        "instance_type": "ml.t2.medium",
        "instance_count": 1
    }
)

print(f"Model deployed to SageMaker and endpoint '{endpoint_name}' created.")
