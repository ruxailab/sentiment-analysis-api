# Huggingface Transformers Parent Image
# FROM huggingface/transformers-pytorch-cpu:4.18.0
# FROM mapler/pytorch-cpu
FROM cnstark/pytorch:2.0.1-py3.10.11-ubuntu22.04

# Set the working directory to /sentiment_analysis_app
WORKDIR /sentiment_analysis_app



# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# RUN pip install --trusted-host pypi.python.org -r requirements.txt
RUN pip install -r requirements.txt

# install ffmpeg
RUN apt-get -y update && apt-get -y upgrade && apt-get install -y --no-install-recommends ffmpeg


# Copy the current directory contents into the container at /sentiment_analysis_app
ADD app/app.py /sentiment_analysis_app/app/app.py
# ADD demos /sentiment_analysis_app/demos

# Copy the current directory contents into the container at /sentiment_analysis_app
ADD ai/src/sentiment_analysis/models/roberta_sentiment.py /sentiment_analysis_app/ai/src/sentiment_analysis/models/roberta_sentiment.py
ADD ai/src/sentiment_analysis/inference/inference.py /sentiment_analysis_app/ai/src/sentiment_analysis/inference/inference.py

ADD ai/src/video_processing /sentiment_analysis_app/ai/src/video_processing

# Make port 8000 available to the world outside this container
EXPOSE 8000

# # Define environment variable


# Run app.py when the container launches
CMD ["python", "/sentiment_analysis_app/app/app.py"]