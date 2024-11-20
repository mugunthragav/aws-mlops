import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import boto3
import os

# Initialize S3 client
s3 = boto3.client('s3')
data_bucket = 'data-bucket-house-raw-data'
processed_bucket = 'data-bucket-house-processed-data'

def load_data():
    # Download dataset from S3
    s3.download_file(data_bucket, 'raw/Housing.csv', 'Housing.csv')

    # Load and preprocess dataset
    df = pd.read_csv('Housing.csv')
    df = pd.get_dummies(df, drop_first=True)
    return df

def preprocess_data(df):
    # Manually encode 'furnishingstatus' to ensure a single column
    df['furnishingstatus'] = df['furnishingstatus'].map({
        'furnished': 2,
        'semi-furnished': 1,
        'unfurnished': 0
    })

    # Ensuring only features are included
    X = df.drop(columns=['price'])
    y = df['price']

    # Split data and scale features
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save processed data and scaler
    processed_data_dir = 'data/processed'
    os.makedirs(processed_data_dir, exist_ok=True)
    pd.DataFrame(X_train_scaled, columns=X.columns).to_csv(os.path.join(processed_data_dir, 'X_train.csv'), index=False)
    pd.DataFrame(X_test_scaled, columns=X.columns).to_csv(os.path.join(processed_data_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(processed_data_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(processed_data_dir, 'y_test.csv'), index=False)
    joblib.dump(scaler, os.path.join(processed_data_dir, 'scaler.pkl'))

    # Upload processed data to S3
    s3.upload_file(os.path.join(processed_data_dir, 'X_train.csv'), processed_bucket, 'X_train.csv')
    s3.upload_file(os.path.join(processed_data_dir, 'X_test.csv'), processed_bucket, 'X_test.csv')
    s3.upload_file(os.path.join(processed_data_dir, 'y_train.csv'), processed_bucket, 'y_train.csv')
    s3.upload_file(os.path.join(processed_data_dir, 'y_test.csv'), processed_bucket, 'y_test.csv')
    s3.upload_file(os.path.join(processed_data_dir, 'scaler.pkl'), processed_bucket, 'scaler.pkl')

    print("Data preprocessing completed.")

if __name__ == "__main__":
    df = load_data()
    preprocess_data(df)
