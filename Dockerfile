# Use a more specific and official base image
FROM python:3.11-slim

# Install CA certificates
RUN apt-get update && apt-get install -y ca-certificates

# Set environment variables to prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    wget \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create and set working directory
WORKDIR /app

# Copy and install dependencies in a single step
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the desired port
EXPOSE 8501
EXPOSE 11434

# Define the entrypoint for the Streamlit app
ENTRYPOINT ["sh", "-c", "streamlit run app.py --server.port=8501 --server.address=0.0.0.0"]

