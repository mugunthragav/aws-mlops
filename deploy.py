import mlflow
import mlflow.deployments
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

    # Get the latest model version in 'Production' stage from the Model Registry
    client = MlflowClient()
    versions = client.get_latest_versions(model_name, stages=["Production"])
    if not versions:
        raise ValueError(f"No version found for model '{model_name}' in stage 'Production'")

    latest_version = versions[0].version
    model_uri = f"models:/{model_name}/{latest_version}"

    print(f"Deploying model from {model_uri} to SageMaker...")

    # Replace <ECR-URL> with your actual ECR URL in the format {account_id}.dkr.ecr.{region}.amazonaws.com/{repo_name}:{tag}
    image_ecr_url = "207567773639.dkr.ecr.us-east-1.amazonaws.com/mlflow-pyfunc"  # Replace with your actual ECR URL

    try:
        # Get the deployment client for SageMaker
        deployment_client = mlflow.deployments.get_deploy_client("sagemaker:/" + region)

        # Create the deployment on SageMaker
        deployment_client.create_deployment(
            name=model_name,
            model_uri=model_uri,
            
            config={
                "image_url": image_ecr_url,
                "execution_role_arn"=execution_role_arn# Provide the ECR URL of your custom image
            }
        )

        print("Model deployed successfully!")
    except Exception as e:
        print(f"Error deploying model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    deploy_model()
