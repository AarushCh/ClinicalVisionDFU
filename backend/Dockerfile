# Use the official Python image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire backend directory into the container
COPY . .

# Expose the port that FastAPI runs on (Hugging Face routes to port 7860 by default)
EXPOSE 7860

# Command to run the application using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
