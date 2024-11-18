import boto3
import mlflow.sagemaker as mfs
from mlflow.tracking import MlflowClient
import mlflow

def deploy_model():
    model_name = "HousePricePrediction01"
    stage = "Production"  # The model stage you want to deploy
    region = "us-east-1"   # Specify the AWS region

    # Set the MLflow tracking URI (adjust this as needed)
    mlflow.set_tracking_uri("http://ec2-100-24-6-128.compute-1.amazonaws.com:5000")  # Replace with your actual MLflow URI

    # Initialize MLflow Client
    client = MlflowClient()

    # Step 1: Ensure the registered model exists
    try:
        registered_model = client.get_registered_model(model_name)
        print(f"Registered Model Info: {registered_model}")
    except Exception as e:
        print(f"Error retrieving registered model: {str(e)}")
        return

    # Step 2: Retrieve the latest model version in the 'Production' stage
    try:
        model_versions = client.get_latest_versions(model_name, stages=[stage])
        if not model_versions:
            raise ValueError(f"No model version found for {model_name} in stage {stage}")
        
        latest_model_version = model_versions[0].version
        print(f"Using model version: {latest_model_version}")
        
    except Exception as e:
        print(f"Error retrieving model version: {str(e)}")
        return

    # Step 3: Define SageMaker deployment parameters
    endpoint_name = "house-price-prediction-endpoint-01"  # SageMaker endpoint name
    execution_role_arn = "arn:aws:iam::207567773639:role/service-role/aws-mlflow"  # Replace with your IAM role ARN
    instance_type = "ml.t2.medium"  # Choose the instance type for deployment

    # Step 4: Deploy the model to SageMaker using the correct MLflow deployment function
    try:
        model_uri = f"models:/{model_name}/{latest_model_version}"  # Correct model URI based on version
        print(f"Model URI: {model_uri}")  # Print model URI for debugging

        # Deploy the model using MLflow's SageMaker deploy function
        mfs._deploy(
            model_uri=model_uri,
            region_name=region,
            execution_role_arn=execution_role_arn,  # IAM Role for SageMaker
            instance_type=instance_type,
            endpoint_name=endpoint_name,  # Specify the endpoint name
            mode="replace"  # Replace the existing endpoint if one exists
        )

        print(f"Model successfully deployed to SageMaker with endpoint: {endpoint_name}")
        print(f"Endpoint URL: https://runtime.sagemaker.{region}.amazonaws.com/endpoints/{endpoint_name}/invocations")
        
    except Exception as e:
        print(f"Deployment failed due to error: {str(e)}")

if __name__ == "__main__":
    deploy_model()
