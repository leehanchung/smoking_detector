#!/bin/bash

pip freeze > api/requirements.txt
sed -i -e 's/-gpu//g' api/requirements.txt
docker build -t smoking_detector_img -f api/Dockerfile .

