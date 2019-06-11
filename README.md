# Smoking Detector

Takes an URL to a video (e.g., Youtube), detect frames that contain persons
smoking, and outputs images with persons smoking in bounding boxes for proof.  


# Environment Setup

Using pipenv.

```
pipenv run python smoking_detector
```

# Building Docker image
```
./tasks/build_api_docker.sh
```

# Testing
With bare python
```
python api/app.py
```

or via Docker
```
tasks/run_docker.sh
```

To test smoking detection REST API, first set the API_URL
```
export API_URL=http://0.0.0.0:8000
```
Then we can test the `GET` method.
```
curl "${API_URL}/smoking_detect?image_url=http://farm8.staticflickr.com/7450/9591155503_4a60f3e1d2_z.jpg"
```
It should return `{"class:":"[0]"}` as the name of the object and the confidence percentage.

## Project structure

Web backend

```
api/                            # Code for serving predictions as a REST API.
__  init__.py
    cli_app.py                  # Command line app that serves predictions without flask
    app.py                      # Flask web server that serves the predictions
    tests/test_app.py           # Integration test for app.py
    Dockerfile                  # Specifies Docker image that runs the web server.
    serverless.yml              # Specifies Serverless framework for AWS Lambda deployment
```

Data (not under version control - one level up in the heirarchy) # NOT IMPLEMENTED

```
data/                            # NOT IMPLEMENTED Training data lives here
    raw/
        emnist/metadata.toml     # Specifications for downloading data
```

Experimentation

```
    evaluation/                     # NOT IMPLEMENTED Scripts for evaluating model on eval set.
        evaluate_character_predictor.py

    notebooks/                  # For snapshots of initial exploration, before solidfying code as proper     1_tensorflow_object_detection_api_demo_colab.ipynb  # tensorflow object detecion api on colab

```

Convenience scripts

```
    tasks/
        # Deployment
        build_api_docker.sh
        deploy_api_to_lambda.sh

        # Tests
        run_prediction_tests.sh
        run_validation_tests.sh
        test_api.sh

        # Training
        train_character_predictor.sh
```

Main model and training code # NOT IMPLEMENTED

```
    text_recognizer/                # Package that can be deployed as a self-contained prediction system
        __init__.py

        character_predictor.py      # Takes a raw image and obtains a prediction
        line_predictor.py

        datasets/                   # Code for loading datasets
            __init__.py
            dataset.py              # Base class for datasets - logic for downloading data
            emnist_dataset.py
            emnist_essentials.json
            dataset_sequence.py

        models/                     # Code for instantiating models, including data preprocessing and loss functions
            __init__.py
            base.py                 # Base class for models
            character_model.py

        networks/                   # Code for building neural networks (i.e., 'dumb' input->output mappings) used by models
            __init__.py
            mlp.py

        tests/
            support/                        # Raw data used by tests
            test_character_predictor.py     # Test model on a few key examples

        weights/                            # Weights for production model
            CharacterModel_EmnistDataset_mlp_weights.h5

        util.py

    training/                       # Code for running training experiments and selecting the best model.
        gpu_util_sampler.py
        run_experiment.py           # Parse experiment config and launch training.
        util.py                     # Logic for training a model with a given config
```
