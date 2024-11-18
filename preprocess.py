import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import boto3

# Initialize S3 client
s3 = boto3.client('s3')
data_bucket = 'data-bucket-house-raw-data'
processed_bucket = 'data-bucket-house-processed-data'

def preprocess_data():
    # Download dataset from S3
    s3.download_file(data_bucket, 'raw/Housing.csv', 'Housing.csv')

    # Load and preprocess dataset
    df = pd.read_csv('Housing.csv')
    categorical_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea','furnishingstatus']
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    X = df.drop('price', axis=1)
    y = df['price']

    # Split data and scale features
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save processed data and scaler
    joblib.dump((X_train_scaled, y_train), 'X_train.pkl')
    joblib.dump((X_test_scaled, y_test), 'X_test.pkl')
    joblib.dump(scaler, 'scaler.pkl')

    # Upload to S3
    s3.upload_file('X_train.pkl', processed_bucket, 'X_train.pkl')
    s3.upload_file('X_test.pkl', processed_bucket, 'X_test.pkl')
    s3.upload_file('scaler.pkl', processed_bucket, 'scaler.pkl')

    print("Data preprocessing completed.")

if __name__ == "__main__":
    preprocess_data()
