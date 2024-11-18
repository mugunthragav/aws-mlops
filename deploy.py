import boto3
import mlflow.sagemaker as mfs
from mlflow.tracking import MlflowClient

def deploy_model():
    model_name = "HousePricePrediction01"
    stage = "Production"  # The model stage you want to deploy
    region = "us-east-1"   # Specify the AWS region

    # Initialize MLflow Client
    client = MlflowClient()

    # Step 1: Retrieve the registered model and its versions
    try:
        # Retrieve the model's registered versions
        registered_model = client.get_registered_model(model_name)
        print(f"Registered Model Info: {registered_model}")
        
        # Retrieve the latest version of the model in the "Production" stage
        model_versions = client.get_latest_versions(model_name, stages=[stage])
        if not model_versions:
            raise ValueError(f"No model version found for {model_name} in stage {stage}")
        
        latest_model_version = model_versions[0].version
        print(f"Using model version: {latest_model_version}")
        
    except Exception as e:
        print(f"Error retrieving model version: {str(e)}")
        return

    # Step 2: Define SageMaker deployment parameters
    app_name = "house-price-prediction-endpoint-01"
    execution_role_arn = "arn:aws:iam::207567773639:role/service-role/aws-mlflow"  # Replace with your IAM role ARN
    instance_type = "ml.t2.medium"  # Choose the instance type for deployment
    instance_count = 1  # Number of instances for deployment

    # Step 3: Deploy the model to SageMaker
    try:
        model_uri = f"models:/{model_name}/{latest_model_version}"  # Correct model URI based on version
        print(f"Model URI: {model_uri}")  # Print model URI for debugging

        # Deploy model using MLflow SageMaker integration
        mfs.push_model_to_sagemaker(
            model_uri=model_uri,
            app_name=app_name,
            region_name=region,
            execution_role_arn=execution_role_arn,
            instance_type=instance_type,
            instance_count=instance_count,
            mode="replace"  # Replace the existing endpoint if one exists
        )

        print(f"Model successfully deployed to SageMaker with endpoint: {app_name}")
        print(f"Endpoint URL: https://runtime.sagemaker.{region}.amazonaws.com/endpoints/{app_name}/invocations")
        
    except Exception as e:
        print(f"Deployment failed due to error: {str(e)}")

if __name__ == "__main__":
    deploy_model()
