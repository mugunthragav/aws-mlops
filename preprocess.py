import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
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

    # Load dataset
    df = pd.read_csv('Housing.csv')
    return df

def preprocess_data(df):
    # Define features and target
    X = df.drop(columns=['price'])
    y = df['price']

    # Define categorical and numerical columns
    categorical_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']
    numerical_columns = [col for col in X.columns if col not in categorical_columns]

    # Preprocessing for numerical data: scaling
    numerical_transformer = StandardScaler()

    # Preprocessing for categorical data: one-hot encoding
    categorical_transformer = OneHotEncoder(drop='first')

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_columns),
            ('cat', categorical_transformer, categorical_columns)
        ]
    )

    # Create and apply pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
    X_processed = pipeline.fit_transform(X)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

    # Save processed data and pipeline
    processed_data_dir = 'data/processed'
    os.makedirs(processed_data_dir, exist_ok=True)
    pd.DataFrame(X_train).to_csv(os.path.join(processed_data_dir, 'X_train.csv'), index=False)
    pd.DataFrame(X_test).to_csv(os.path.join(processed_data_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(processed_data_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(processed_data_dir, 'y_test.csv'), index=False)
    joblib.dump(pipeline, os.path.join(processed_data_dir, 'pipeline.pkl'))

    # Upload processed data to S3
    s3.upload_file(os.path.join(processed_data_dir, 'X_train.csv'), processed_bucket, 'X_train.csv')
    s3.upload_file(os.path.join(processed_data_dir, 'X_test.csv'), processed_bucket, 'X_test.csv')
    s3.upload_file(os.path.join(processed_data_dir, 'y_train.csv'), processed_bucket, 'y_train.csv')
    s3.upload_file(os.path.join(processed_data_dir, 'y_test.csv'), processed_bucket, 'y_test.csv')
    s3.upload_file(os.path.join(processed_data_dir, 'pipeline.pkl'), processed_bucket, 'pipeline.pkl')

    print("Data preprocessing completed.")

if __name__ == "__main__":
    df = load_data()
    preprocess_data(df)
