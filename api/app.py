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

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend

# from smoking_detector.line_predictor import LinePredictor
sys.path.append("..")
# from tensorflow.models.research.object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.13.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.13.*.')

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


def _detect():
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
        print('GET')
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


# INPUT: image in PIL format
# OUTPUT: image in np format.
def _load_image_into_numpy_array(image):##, dim=4):
  (im_width, im_height) = image.size
  #if dim == 4:
  return np.array(image.getdata()).reshape(
        (1, im_height, im_width, 3)).astype(np.uint8)
"""  elif dim == 3:
      return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)
"""

@app.route('/smoking_detect', methods=['GET'])
def smoking_detection():
    #response = requests.get(url)
    image_np = _load_image()
    model = resnet()
    model.load_weights('../smoking_detector/weights/latest_model_weights.h5')
    pred = model.predict_classes(image_np)
    #percent = 0

    return jsonify({'class:': str(pred)})#, 'percent:': float(perc)})


@app.route('/obj_detect', methods=['GET'])#, 'POST'])
def object_detection():

    image_np = _load_image()
    category_index = _load_label(PATH_TO_LABELS)
    detection_graph = _load_model(tf.Graph(), PATH_TO_FROZEN_GRAPH)
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)

            top_detected_class = category_index[np.squeeze(classes).astype(np.int32)[0]]['name']
            top_detected_perc = np.squeeze(scores)[0]
            # Transforming image_np back to PIL format for display
            #detected_image = Image.fromarray(image_np, 'RGB')
            #detected_image.show()
            #class_name = category_index[classes[i]]['name']
            #return jsonify({'pred': str(pred), 'conf': float(conf)})
    return jsonify({'class:': str(top_detected_class), 'percent:': float(top_detected_perc)})
    #return "{}: {:.2%}".format(category_index[np.squeeze(classes).astype(np.int32)[0]]['name'], np.squeeze(scores)[0])


def main():
    app.run(host='0.0.0.0', port=8000, debug=False)  # nosec

if __name__ == '__main__':
    main()
