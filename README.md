# Smoking Detector

Takes an URL to a video (e.g., Youtube), detect frames that contain persons
smoking, and outputs images with persons smoking in bounding boxes for proof.  


# Environment Setup

Using pipenv.

```
pipenv run python smoking_detector
```

## Project structure

Web backend

```
api/                        # Code for serving predictions as a REST API.
    tests/test_app.py           # NOT YET IMPLEMENTED Test that predictions are working
    Dockerfile                  # NOT YET IMPLEMENTED Specificies Docker image that runs the web server.
    __init__.py
    app.py                      # NOT YET IMPLEMENTED Flask web server that serves predictions.
    mockup.py                   # app.py but running locally without Flask
    serverless.yml              # NOT YET IMPLEMENTED Specifies AWS Lambda deployment of the REST API.
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

Convenience scripts # NOT IMPLEMENTED

```
    tasks/
        # Deployment
        build_api_docker.sh
        deploy_api_to_lambda.sh

        # Code quality
        lint.sh

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
