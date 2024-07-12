# Start with the Ubuntu 20.04 base image
FROM ubuntu:20.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install Python, pip, nano, iputils-ping and iproute2 (for network management and troubleshooting)
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3-pip \
    nano \
    iputils-ping \
    iproute2 \
    && rm -rf /var/lib/apt/lists/*

# Update pip and install required Python packages
RUN python3.8 -m pip install --upgrade pip && \
    python3.8 -m pip install pandas scikit-learn nvflare torch torchvision xgboost

# Set the working directory to /workspace
WORKDIR /workspace
