# Start from the latest Long Term Support (LTS) Ubuntu version
# FROM tensorflow/tensorflow:latest-py3
FROM python:3.6
# FROM ubuntu:18.04
# FROM amazon/aws-eb-python:3.4.2-onbuild-3.5.1

# Install pipenv
RUN apt-get update && apt-get install python3-pip -y && pip3 install pipenv
RUN apt-get install -y libsm6 libxext6 libxrender-dev

# Create the working directory
RUN set -ex && mkdir /repo
WORKDIR /repo

# Copy only the relevant directories to the working diretory
COPY smoking_detector/ ./smoking_detector
COPY api/ ./api

# Install Python dependencies
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
RUN set -ex && pip3 install -r api/requirements.txt
# RUN set -ex && pip3 install opencv-python==4.1.0.25
# RUN set -ex && pip3 install youtube-dl==2019.5.20

# Run the web server
EXPOSE 8000
ENV PYTHONPATH /repo
CMD python3 /repo/api/app.py
