import pandas as pd
from sklearn.model_selection import train_test_split
import boto3
import os

# Initialize S3 client
s3 = boto3.client('s3')
data_bucket = 'data-bucket-house-raw-data'
processed_bucket = 'data-bucket-house-processed-data'

def load_data():
    # Download dataset from S3
    s3.download_file(data_bucket, 'raw/Housing.csv', 'Housing.csv')

    # Load the dataset
    data = pd.read_csv('Housing.csv')
    data = pd.get_dummies(data, drop_first=True)
    return data

def get_train_test_data(data, test_size, random_state):
    # Prepare the train and test data
    X = data.drop('price', axis=1)
    y = data['price']
    print(f'Training data shape: {X.shape}')
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def save_data(X_train, X_test, y_train, y_test):
    # Save processed data to CSV files
    processed_data_dir = 'data/processed'
    os.makedirs(processed_data_dir, exist_ok=True)
    
    X_train.to_csv(os.path.join(processed_data_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(processed_data_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(processed_data_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(processed_data_dir, 'y_test.csv'), index=False)

    # Upload processed data to S3
    s3.upload_file(os.path.join(processed_data_dir, 'X_train.csv'), processed_bucket, 'X_train.csv')
    s3.upload_file(os.path.join(processed_data_dir, 'X_test.csv'), processed_bucket, 'X_test.csv')
    s3.upload_file(os.path.join(processed_data_dir, 'y_train.csv'), processed_bucket, 'y_train.csv')
    s3.upload_file(os.path.join(processed_data_dir, 'y_test.csv'), processed_bucket, 'y_test.csv')

    print("Data saved and uploaded to S3.")

if __name__ == "__main__":
    data = load_data()
    X_train, X_test, y_train, y_test = get_train_test_data(data, test_size=0.2, random_state=42)
    save_data(X_train, X_test, y_train, y_test)
