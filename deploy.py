import mlflow.deployments
import boto3

# AWS configuration
aws_region = 'your_aws_region'
execution_role_arn = 'arn:aws:iam::207567773639:role/service-role/aws-mlflow'  # Replace with your SageMaker execution role ARN
model_name = 'HousePricePrediction01'
model_version = '1'  # Replace with your model version if needed
endpoint_name = 'house-price-prediction-endpoint'  # Replace with desired endpoint name

# MLflow model URI
model_uri = f'models:/{model_name}/{model_version}'

# Deploy model to SageMaker
client = mlflow.deployments.get_deploy_client("sagemaker")

client.create_deployment(
    name=endpoint_name,
    model_uri=model_uri,
    config={
        "execution_role_arn": execution_role_arn,
        "region_name": aws_region,
        "instance_type": "ml.m5.large",
        "instance_count": 1
    }
)

print(f"Model deployed to SageMaker and endpoint '{endpoint_name}' created.")
