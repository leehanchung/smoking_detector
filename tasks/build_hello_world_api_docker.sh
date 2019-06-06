#!/bin/bash

pipenv lock --requirements --keep-outdated > api/requirements.txt
sed -i 's/-gpu//g' api/requirements.txt
docker build -t hello_world_api -f api/Hello_World_Dockerfile .
