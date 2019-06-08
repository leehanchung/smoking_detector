"""Flask web server serving text_recognizer predictions."""
# From https://github.com/UnitedIncome/serverless-python-requirements
# From https://github.com/full-stack-deep-learning/fsdl-text-recognizer-project/blob/master/lab9_sln/api/app.py
# app.py is the actuall app.  it imports smoking_detector's detector models
# to make detections.
# make it print a segmented image online first
try:
    import unzip_requirements  # pylint: disable=unused-import
except ImportError:
    pass

from flask import Flask, request, jsonify
import os, sys
import tarfile, zipfile
from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO

# The download section hangs when the files to be downloaded exists so adding a conditional
import tensorflow as tf
tf.enable_eager_execution()
from pathlib import Path
from typing import Union
import six.moves.urllib as urllib
from urllib.request import urlopen, urlretrieve

sys.path.append("..")
from tensorflow.keras import backend

# from smoking_detector.line_predictor import LinePredictor
sys.path.append("..")
# from tensorflow.models.research.object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.13.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.13.*.')

#
# Import Utils from tensorflow object_detection directory
#
# from tensorflow.models.research.object_detection.utils import label_map_util
# from tensorflow.models.research.object_detection.utils import visualization_utils as vis_util

# from text_recognizer.datasets import IamLinesDataset
# import text_recognizer.util as util

app = Flask(__name__)  # pylint: disable=invalid-name

# Tensorflow bug: https://github.com/keras-team/keras/issues/2397
# with backend.get_session().graph.as_default() as _:
#     predictor = LinePredictor()  # pylint: disable=invalid-name
    # predictor = LinePredictor(dataset_cls=IamLinesDataset)

@app.route('/')
def index():
    return 'Hello, world!'


@app.route('/v1/predict', methods=['GET', 'POST'])
def predict():
    image = _load_image()
    with backend.get_session().graph.as_default() as _:
        pred, conf = (0.0002, 0.0002) #predictor.predict(image)
        print("meanTRIC confidence {}".format(conf))
        print("METRIC mean_intensity {}".format(2.2)) # image.mean()
        print("INFO pred {}".format(pred))
    return jsonify({'pred': str(pred), 'conf': float(conf)})


def detect():
    return None


def _load_video_url():
    return None


def _load_image():
    if request.method == 'POST':
        data = request.get_json()
        if data is None:
            return 'no json received'
        return read_b64_image(data['image'], grayscale=True)
    if request.method == 'GET':
        image_url = request.args.get('image_url')
        if image_url is None:
            return 'no image_url defined in query string'
        print("INFO url {}".format(image_url))
        return read_image(image_url, grayscale=True)
    raise ValueError('Unsupported HTTP method')

def read_b64_image(b64_string, grayscale=False):
    return None
    # """Load base64-encoded images."""
    # import base64
    # imread_flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    # try:
    #     _, b64_data = b64_string.split(',')
    #     return cv2.imdecode(np.frombuffer(base64.b64decode(b64_data), np.uint8), imread_flag)
    # except Exception as e:
    #     raise ValueError("Could not load image from b64 {}: {}".format(b64_string, e))

def read_image(image_uri: Union[Path, str], grayscale=False) -> tf.Tensor:
    """Read image_uri."""

    def read_image_from_url(image_url):
        url_response = urlopen(str(image_url))  # nosec
        image = tf.image.decode_image(url_response.read())
        print(f"tf.Tensor.shape {image.shape}")
        return image

    try:
        # img = None
        img = read_image_from_url(image_uri)
        assert img is not None
    except Exception as e:
        raise ValueError("Could not load image at {}: {}".format(image_uri, e))
    return img


def main():
    app.run(host='0.0.0.0', port=8000, debug=False)  # nosec

if __name__ == '__main__':
    main()
