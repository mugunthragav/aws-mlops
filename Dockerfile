# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file into the container
COPY requirements.txt .

# Install Python dependencies (including MLflow and any necessary libraries)
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose necessary ports (e.g., MLflow, Streamlit)
EXPOSE 5000


# Default command to run deploy script (this will run when container starts)
CMD ["python", "deploy.py"]
