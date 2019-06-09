"""Flask web server serving smoking_detector predictions."""
# From https://github.com/UnitedIncome/serverless-python-requirements
# From https://github.com/full-stack-deep-learning/fsdl-text-recognizer-project/blob/master/lab9_sln/api/app.py
# app.py is the actuall app.  it imports smoking_detector's detector models
# to make detections.

try:
    import unzip_requirements  # pylint: disable=unused-import
except ImportError:
    pass

from flask import Flask, request, jsonify
# Must compile and install opencv-python, won't work if pip install opencv-python
# Video processing libraries
import cv2, pafy, youtube_dl
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend

from distutils.version import StrictVersion
if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')

# Import Utils from tensorflow object_detection directory
from smoking_detector.utils.utils import ops as utils_ops
from smoking_detector.utils.utils import label_map_util
from smoking_detector.utils.utils import visualization_utils as vis_util
import smoking_detector.utils.util as util

PATH_TO_FROZEN_GRAPH = 'smoking_detector/weights/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb'
PATH_TO_LABELS = 'smoking_detector/datasets/mscoco_label_map.pbtxt'

app = Flask(__name__)  # pylint: disable=invalid-name

# Tensorflow bug: https://github.com/keras-team/keras/issues/2397
#with backend.get_session().graph.as_default() as _:
#    predictor = LinePredictor()  # pylint: disable=invalid-name
    # predictor = LinePredictor(dataset_cls=IamLinesDataset)

@app.route('/')
def index():
    return 'Hello, world!'


@app.route('/v1/predict', methods=['GET', 'POST'])
def predict():
    image = _load_image()
    with backend.get_session().graph.as_default() as _:
        pred, conf = predictor.predict(image)
        print("METRIC confidence {}".format(conf))
        print("METRIC mean_intensity {}".format(image.mean()))
        print("INFO pred {}".format(pred))
    return jsonify({'pred': str(pred), 'conf': float(conf)})


def _detect():
    return None

def _load_model(graph, model):
    with graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(model, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    return graph

def _load_label(label_file):
    #category_index =
    return label_map_util.create_category_index_from_labelmap(label_file, use_display_name=True)


def _load_video_url():
    return None


def _load_image():
    if request.method == 'POST':
        print('POST')
        data = request.get_json()
        if data is None:
            return 'no json received'
        return util.read_b64_image(data['image'])#, grayscale=True)
    if request.method == 'GET':
        print('GET')
        image_url = request.args.get('image_url')
        if image_url is None:
            return 'no image_url defined in query string'
        print("INFO url {}".format(image_url))
        return util.read_image(image_url)#, grayscale=True)
    raise ValueError('Unsupported HTTP method')


# INPUT: image in PIL format
# OUTPUT: image in np format.
def _load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


@app.route('/obj_detect', methods=['GET', 'POST'])
def image_detection():
    #image_file = 'smoking_detector/datasets/image1.jpg'#_load_image()
    """image_np = cv2.imread(image_file)
    #image = Image.open(image_file)
    cv2.imshow(image_np)#image.show()
    image_np = _load_image_into_numpy_array(image)
    """
    #image_np = util.read_image(image_file)
    image_np = _load_image()  ## BUG.  IMDECODE returns None.  urlimg cant be decoded?!?!?
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
