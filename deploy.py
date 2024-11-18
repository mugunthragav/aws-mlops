import boto3
import mlflow.sagemaker as mfs

def deploy_model():
    model_name = "HousePricePrediction01"
    stage = "Production"
    region = "us-east-1"

    # Deploy to SageMaker
    app_name = "house-price-prediction-endpoint-01"
    mfs.deploy(
        model_uri=f"models:/{model_name}/{stage}",
        app_name=app_name,
        region_name=region,
        mode="replace"
    )
    print(f"Model deployed to SageMaker with endpoint: {app_name}")

if __name__ == "__main__":
    deploy_model()
