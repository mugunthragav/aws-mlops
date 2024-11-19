import mlflow
import boto3
import joblib
from mlflow.exceptions import RestException

# Initialize S3 client
s3 = boto3.client('s3')
model_bucket = 'model-bucket-house-model'

def register_and_promote():
    # Download best model from S3
    s3.download_file(model_bucket, 'best_model.pkl', 'best_model.pkl')
    model = joblib.load('best_model.pkl')

    # Load the best model name
    with open('best_model_name.txt', 'r') as f:
        best_model_name = f.read().strip()

    # Initialize MLflow client
    mlflow.set_tracking_uri("http://ec2-100-24-6-128.compute-1.amazonaws.com:5000")
    client = mlflow.tracking.MlflowClient()

    # Register the model if it doesn't exist
    try:
        client.get_registered_model(best_model_name)
        print(f"Model {best_model_name} is already registered.")
    except RestException as e:
        if "RESOURCE_DOES_NOT_EXIST" in str(e):
            print(f"Model {best_model_name} is not registered. Registering now.")
            mlflow.sklearn.log_model(model, best_model_name, registered_model_name=best_model_name)
        else:
            raise

    # Get the latest version of the model
    versions = client.get_latest_versions(best_model_name, stages=["Production"])
    if versions:
        latest_version = versions[0].version
        print(f"Latest version of {best_model_name}: {latest_version}")
    else:
        print(f"No versions found for {best_model_name} in the 'Production' stage.")
        return  # Exit if no version is found

    # Transition model to production
    client.transition_model_version_stage(
        name=best_model_name,
        version=latest_version,
        stage="Production"
    )
    print(f"Model {best_model_name} version {latest_version} promoted to Production.")

if __name__ == "__main__":
    register_and_promote()
