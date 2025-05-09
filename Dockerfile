# Use NVIDIA's CUDA-enabled Python image for GPU support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Install Python 3.12 and distutils
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3.12 python3.12-venv python3.12-dev python3.12-distutils python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Update pip to the latest version to avoid dependency resolution issues
RUN python3.12 -m pip install --no-cache-dir --upgrade pip

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN python3.12 -m pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Command to run the application (modify as needed)
CMD ["python3.12", "main.py"]