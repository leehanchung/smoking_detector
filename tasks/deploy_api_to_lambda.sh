#!/bin/bash

pip freeze > api/requirements.txt
sed -i -e 's/tensorflow-gpu/tensorflow/' api/requirements.txt
cd api || exit 1
npm install
sls deploy -v
