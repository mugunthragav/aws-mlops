import boto3
import mlflow.sagemaker as mfs
from mlflow.tracking import MlflowClient

def deploy_model():
    model_name = "HousePricePrediction01"
    stage = "Production"
    region = "us-east-1"

    # Check if the model exists in the registry
    client = MlflowClient()
    try:
        model_versions = client.get_latest_versions(model_name, stages=[stage])
        if not model_versions:
            raise ValueError(f"No model version found for {model_name} in stage {stage}")
        latest_model_version = model_versions[0].version
        print(f"Using model version: {latest_model_version}")
    except Exception as e:
        print(f"Error checking model version: {str(e)}")
        return

    # Deploy to SageMaker
    app_name = "house-price-prediction-endpoint-01"
    execution_role_arn = "arn:aws:iam::YOUR_AWS_ACCOUNT_ID:role/YOUR_SAGEMAKER_EXECUTION_ROLE"  # Replace with your IAM role ARN
    instance_type = "ml.m5.large"
    instance_count = 1

    try:
        # Deploy the model to SageMaker
        mfs.push_model_to_sagemaker(
            model_uri=f"models:/{model_name}/{stage}",  # Ensure the stage is correct
            app_name=app_name,
            region_name=region,
            execution_role_arn=execution_role_arn,
            instance_type=instance_type,
            instance_count=instance_count,
            mode="replace"
        )
        print(f"Model successfully deployed to SageMaker with endpoint: {app_name}")
        print(f"Endpoint URL: https://runtime.sagemaker.{region}.amazonaws.com/endpoints/{app_name}/invocations")
    except Exception as e:
        print(f"Deployment failed due to error: {str(e)}")

if __name__ == "__main__":
    deploy_model()
