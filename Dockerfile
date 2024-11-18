# Base image for MLflow PyFunc
FROM continuumio/miniconda3:4.12.0

# Set working directory
WORKDIR /mlflow

# Install MLflow and required dependencies
RUN conda install -y python=3.8 && \
    pip install mlflow==2.17.2 boto3 && \
    conda clean -a

# Expose default MLflow ports
EXPOSE 5000

# Set the entrypoint to MLflow
ENTRYPOINT ["mlflow"]
