# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.9-slim

# Set the working directory in the container to /GPTModelInferenceAPI
WORKDIR /GPTModelInferenceAPI

# Copy the current directory contents into the container at /GPTModelInferenceAPI
COPY . /GPTModelInferenceAPI

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Define environment variable
ENV FLASK_APP=gpt_model_inference_api.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=8080

# Run flask application
CMD ["flask", "run"]
