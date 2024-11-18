# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install MLflow
RUN pip install mlflow

# Expose port 5000 for MLflow's UI (optional, if you want to run MLflow UI)
EXPOSE 5000

# Run the MLflow server or your script (use the command for your use case)
CMD ["mlflow", "server", "--host", "0.0.0.0"]
