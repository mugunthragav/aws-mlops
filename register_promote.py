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
    s3.download_file(model_bucket, 'best_model_name.txt', 'best_model_name.txt')
    with open('best_model_name.txt', 'r') as f:
        best_model_name = f.read().strip()

    # Initialize MLflow client
    mlflow.set_tracking_uri("http://ec2-100-24-6-128.compute-1.amazonaws.com:5000")
    client = mlflow.tracking.MlflowClient()

    # Start a new MLflow run
    with mlflow.start_run() as run:
        # Register the model as a new version
        try:
            model_uri = f"runs:/{run.info.run_id}/model"
            mlflow.sklearn.log_model(model, "model", registered_model_name=best_model_name)
            print(f"Model {best_model_name} registered successfully.")
        except RestException as e:
            print(f"Failed to register model {best_model_name}: {str(e)}")
            return

        # Get the latest version of the model
        try:
            versions = client.get_latest_versions(best_model_name, stages=["None"])
            latest_version = max(int(version.version) for version in versions)
            print(f"Latest version of {best_model_name}: {latest_version}")
        except RestException as e:
            print(f"Failed to retrieve latest model versions for {best_model_name}: {str(e)}")
            return

        # Transition latest model version to production
        try:
            client.transition_model_version_stage(
                name=best_model_name,
                version=latest_version,
                stage="Production",
                archive_existing_versions=True
            )
            print(f"Model {best_model_name} version {latest_version} promoted to Production.")
        except RestException as e:
            print(f"Failed to transition model {best_model_name} version {latest_version} to Production: {str(e)}")

if __name__ == "__main__":
    register_and_promote()
