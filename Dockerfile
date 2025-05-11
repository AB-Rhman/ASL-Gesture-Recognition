# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# List contents of the current directory (for debugging)
RUN ls -la

# Copy requirements file first (for better caching)
COPY requirements.txt /app/requirements.txt

# Verify requirements.txt exists
RUN ls -la /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the project files
COPY . /app/

# Create necessary directories
RUN mkdir -p /app/models /app/templates

# List contents after copying (for debugging)
RUN ls -la /app/

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"] 