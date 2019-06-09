#!/bin/bash

#pipenv lock --requirements --keep-outdated > api/requirements.txt
sed -i 's/-gpu//g' api/requirements.txt
docker build -t smoking_detector_api -f api/Dockerfile .

