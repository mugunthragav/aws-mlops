import mlflow
import mlflow.sagemaker as mfs
from mlflow.tracking import MlflowClient

def deploy_model():
    model_name = "HousePricePrediction01"
    app_name = "house-price-predictor"
    region = "us-west-2"
    execution_role_arn = "arn:aws:iam::207567773639:role/sagemakerops"
    instance_type = "ml.t2.medium"

    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://ec2-100-24-6-128.compute-1.amazonaws.com:5000")

    # Get the latest model version in 'Production'
    client = MlflowClient()
    versions = client.get_latest_versions(model_name, stages=["Production"])
    if not versions:
        raise ValueError(f"No version found for model '{model_name}' in stage 'Production'")

    latest_version = versions[0].version
    model_uri = f"models:/{model_name}/{latest_version}"

    print(f"Deploying model from URI: {model_uri}")

    # Use _deploy to deploy the model
    try:
        mfs._deploy(
            app_name=app_name,
            model_uri=model_uri,
            region_name=region,
            execution_role_arn=execution_role_arn,
            instance_type=instance_type,
            instance_count=1,
            mode="create"
        )
        print(f"Model successfully deployed as {app_name} on SageMaker.")
    except Exception as e:
        print(f"Failed to deploy model: {e}")

if __name__ == "__main__":
    deploy_model()
