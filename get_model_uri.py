import mlflow
from mlflow.tracking import MlflowClient
mlflow.set_tracking_uri("http://ec2-100-24-6-128.compute-1.amazonaws.com:5000") 

def get_model_uri():
    model_name = "HousePricePrediction01"
    model_version = 1  # Replace with your model version
    
    client = MlflowClient()
    model_uri = f"models:/{model_name}/{model_version}"
    
    return model_uri

if __name__ == "__main__":
    model_uri = get_model_uri()
    with open('model_uri.txt', 'w') as f:
        f.write(model_uri)
