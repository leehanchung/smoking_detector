# Video processing libraries
# Must compile and install opencv-python, won't work if pip install opencv-python
import cv2
import pafy
import youtube_dl
import numpy as np

# Tensorflow and model reading libraries
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from PIL import Image
# The download section hangs when the files to be downloaded exists so adding a conditional
import pathlib

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')

#
# Import Utils from tensorflow object_detection directory
#
from utils import label_map_util
from utils import visualization_utils as vis_util

#
# Path variables
#
# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
#MODEL_NAME = 'faster_rcnn_resnet101_coco_2017_11_08'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
#print(DOWNLOAD_BASE+MODEL_FILE)
#http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = '../weights/' + MODEL_NAME + '/frozen_inference_graph.pb'
print(PATH_TO_FROZEN_GRAPH)
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('../dataset/', 'mscoco_label_map.pbtxt')
print(PATH_TO_LABELS)

#
# Download model
#
print(pathlib.Path(MODEL_FILE))
downloaded_model = pathlib.Path(MODEL_FILE)
if not downloaded_model.exists():
    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    print(1)
    tar_file = tarfile.open(MODEL_FILE)
    print(2)
    for file in tar_file.getmembers():
        print(3)
        file_name = os.path.basename(file.name)
        if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, os.getcwd())

#
# Load model
#
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

#
# load label Map
#

#category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
category_index = label_map_util.create_category_index_from_labelmap('mscoco_label_map.pbtxt', use_display_name=True)

#
# Helper code
#
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
                ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[1], image.shape[2])
                detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict, feed_dict={image_tensor: image})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.int64)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict

#
# Load Videos
#

url = "https://youtu.be/dQw4w9WgXcQ"

videoPafy = pafy.new(url)
best = videoPafy.getbest(preftype="mp4")

cap = cv2.VideoCapture()
cap.open(best.url)
# Video Cam source
#cap.open(0)

#
# This is very slow. Because it is calling run_inference_for_single_image
# and creating a tf.sess for every image in the video. Instead, in 2streaming_test.py
# we have decided to create only one tf.sess for the whole video and do inference inside
# only one tf.sess, reducing the overhead.
#
while True:
    ret, image_np = cap.read()
    print(ret, image_np)
    #loadedImage=cv2.imdecode(frame, cv2.IMREAD_COLOR)
    #cv2.imshow('frame', loadedImage)

    #your code....

    if image_np is not None:
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        #image_np = load_image_into_numpy_array(image)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=8)

        cv2.imshow('Object Detection', image_np)#cv2.resize(image_np, (800, 600)))
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    else:
        print("NO FRAME")
        break

cap.release()
cv2.destroyAllWindows()
