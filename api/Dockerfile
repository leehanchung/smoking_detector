# Start from the Python 3.6  image
# https://hub.docker.com/_/python
FROM python:3.6

# Install pipenv (alpine uses apk add/update vs ubuntu apt-get
RUN apt-get update && apt-get install python3-pip -y && pip3 install pipenv

#RUN apt-get install -y libsm6 libxext6 libxrender-dev

# Create the working directory
RUN set -ex && mkdir /repo
WORKDIR /repo

# Copy only the relevant directories to the working diretory
COPY smoking_detector/ ./smoking_detector
COPY api/ ./api

# Install Python dependencies
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
RUN set -ex && pip3 install --no-cache-dir -r api/requirements.txt

# Run the web server
EXPOSE 8000
ENV PYTHONPATH /repo
CMD python3 /repo/api/app.py
