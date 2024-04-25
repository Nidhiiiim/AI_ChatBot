# Use an official Python runtime as a parent image
FROM python:3.9.6

# Set the working directory in the container
WORKDIR /usr/src/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libhdf5-dev \
    gcc \
    g++

# Upgrade pip
RUN pip install --upgrade pip

# Copy the current directory contents into the container at /usr/src/app
COPY . /usr/src/app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Define environment variable
ENV NAME World

# Run training_model.py when the container launches
CMD ["python", "./training_model.py"]


