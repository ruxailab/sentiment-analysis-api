# Huggingface Transformers Parent Image
FROM cnstark/pytorch:2.0.1-py3.10.11-ubuntu22.04

# install ffmpeg
RUN apt-get -y update && apt-get -y upgrade && apt-get install -y --no-install-recommends ffmpeg

# Set the working directory to /sentiment_analysis_api
WORKDIR /sentiment_analysis_api

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Make port 8001 available to the world outside this container
EXPOSE 8001

# Run app.py when the container launches
CMD ["python","-u","-m", "run"]