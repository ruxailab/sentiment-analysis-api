# # Use an official Python runtime as a parent image
# FROM python:3.10-slim
# Use the official PyTorch image as a parent image
FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime
# FROM pytorch/pytorch:latest

# Set the working directory to /sentiment_analysis_app
WORKDIR /sentiment_analysis_app

# Copy the requirements file first to leverage Docker cache
COPY app/requirements.txt .


# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt


# Copy the current directory contents into the container at /sentiment_analysis_app
ADD app/app.py /sentiment_analysis_app/app/app.py
ADD app/Basma_sportify_1_Side.mp4 /sentiment_analysis_app/app/Basma_sportify_1_Side.mp4

# Copy the current directory contents into the container at /sentiment_analysis_app
ADD ai/src/sentiment_analysis/models /sentiment_analysis_app/ai/src/sentiment_analysis/models
ADD ai/src/sentiment_analysis/inference /sentiment_analysis_app/ai/src/sentiment_analysis/inference

ADD ai/src/video_processing /sentiment_analysis_app/ai/src/video_processing

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable


# Run app.py when the container launches
CMD ["python", "/sentiment_analysis_app/app/app.py"]


# # # Use the official Python image as a base image
# # FROM python:3.10

# # # Set the working directory in the container
# # WORKDIR /app

# # # Copy the requirements file into the container at /app
# # COPY requirements.txt .

# # # Install the dependencies
# # RUN pip install --no-cache-dir -r requirements.txt

# # # Copy the content of the local src directory to the working directory
# # COPY . .

# # # Expose the port the app runs on
# # EXPOSE 5000

# # # Define environment variable
# # ENV FLASK_APP=app.py

# # # Command to run the Flask app
# # CMD ["flask", "run", "--host=0.0.0.0"]