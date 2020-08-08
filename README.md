# Smoking Detector

Takes an URL to a video (e.g., Youtube), detect frames that contain persons
smoking, and outputs images with persons smoking in bounding boxes for proof.  


# Environment Setup

Using python `venv` module.

```
git clone git@github.com:leehanchung/smoking_detector.git
python -m venv smoking_detector
pip install --upgrade pip
cd smoking_detector
pip install -r api/requirements.txt
```

# Building docker image

```
./tasks/build_api_docker.sh
```

# Testing

With bare python
```
python api/app.py
```

or via docker
```
tasks/run_docker.sh
```

then visit [http://localhost:8000/v1/predict?image_url=A_URL_OF_AN_IMAGE](http://localhost:8000/v1/predict?image_url=)

# Deploying to lambda

```
./tasks/deploy_api_to_lambda.sh
```


## Project structure

Web backend

```
api/                        # Code for serving predictions as a REST API.
    Dockerfile                  #  Specificies Docker image that runs the web server.
    __init__.py
    app.py                      # Flask web server that serves predictions.
    mockup.py                   # app.py but running locally without Flask
    serverless.yml              # Specifies AWS Lambda deployment of the REST API.
```

Convenience scripts 

```
    tasks/
        # Deployment
        build_api_docker.sh
        deploy_api_to_lambda.sh

        # Tests
        run_docker.sh

        # Test left from previous code
        test_hello_world.sh
```

