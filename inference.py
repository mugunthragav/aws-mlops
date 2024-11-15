import boto3
import numpy as np


def invoke_endpoint(endpoint_name, payload):
    runtime = boto3.client("sagemaker-runtime")

    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        Body=payload,
        ContentType="text/csv"
    )
    predictions = response["Body"].read().decode("utf-8")
    return predictions


if __name__ == "__main__":
    # Example: Inference for a single sample
    endpoint_name = "house-price-prediction-endpoint"

    # Replace with actual feature values for inference
    input_features = np.array([[3000, 4, 2, 1, 0, 0, 0, 1]])
    payload = ",".join(map(str, input_features.flatten()))  # Convert to CSV format

    predictions = invoke_endpoint(endpoint_name, payload)
    print("Predicted Price:", predictions)
