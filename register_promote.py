import mlflow
import boto3
import joblib

# Initialize S3 client
s3 = boto3.client('s3')
processed_bucket = 'your-processed-bucket'

def register_and_promote():
    # Download best model
    s3.download_file(processed_bucket, 'best_model.pkl', 'best_model.pkl')
    model = joblib.load('best_model.pkl')

    mlflow.set_tracking_uri("http://your-mlflow-server")
    client = mlflow.tracking.MlflowClient()

    # Register model
    model_name = "HousePricePrediction"
    mlflow.sklearn.log_model(model, model_name)

    # Transition to production
    latest_version = client.get_latest_versions(model_name, stages=["None"])[0].version
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version,
        stage="Production"
    )
    print(f"Model {model_name} version {latest_version} promoted to Production.")

if __name__ == "__main__":
    register_and_promote()
