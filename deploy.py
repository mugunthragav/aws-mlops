import mlflow
import mlflow.sagemaker as mfs
from mlflow.tracking import MlflowClient

def deploy_model():
    # The model registered in your MLflow registry
    model_name = "HousePricePrediction01"  # Name of the model in MLflow registry
    region = "us-west-2"  # AWS region
    execution_role_arn = "arn:aws:iam::207567773639:role/sagemakerops"  # IAM Role ARN for SageMaker

    # Set the MLflow tracking URI (replace with your actual MLflow URI)
    mlflow.set_tracking_uri("http://ec2-100-24-6-128.compute-1.amazonaws.com:5000")  # MLflow URI

    # Initialize the MLflow client
    client = MlflowClient()

    try:
        # Step 1: Retrieve the latest model version in the 'Production' stage
        model_versions = client.get_latest_versions(model_name, stages=["Production"])
        if not model_versions:
            raise ValueError(f"No model version found for {model_name} in stage 'Production'")

        latest_model_version = model_versions[0].version
        print(f"Using model version: {latest_model_version}")

    except Exception as e:
        print(f"Error retrieving model version: {str(e)}")
        return

    try:
        # Step 2: Define the model URI
        model_uri = f"models:/{model_name}/{latest_model_version}"  # URI for the latest version of the model
        print(f"Model URI: {model_uri}")

        # Step 3: Deploy the model to SageMaker
        mfs.push_model_to_sagemaker(
            model_name=model_name,
            model_uri=model_uri,
            execution_role_arn=execution_role_arn,  # IAM Role ARN for SageMaker
            region_name=region,
            vpc_config=None,  # Optional: VPC configuration if needed
            image_url=None,  # Optional: custom image URL if needed
            flavor=None,  # Optional: specify model flavor if needed
        )

        print(f"Model successfully deployed to SageMaker with endpoint: {model_name}-endpoint")
        print(f"Endpoint URL: https://runtime.sagemaker.{region}.amazonaws.com/endpoints/{model_name}-endpoint/invocations")

    except Exception as e:
        print(f"Deployment failed due to error: {str(e)}")

if __name__ == "__main__":
    deploy_model()
