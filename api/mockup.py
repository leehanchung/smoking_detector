# Video processing libraries
# Must compile and install opencv-python, won't work if pip install opencv-python
import cv2, pafy, youtube_dl
import numpy as np

# Tensorflow and model reading libraries
import os, sys, pathlib
import six.moves.urllib as urllib
import tensorflow as tf

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
#from object_detection.utils import ops as utils_ops
from smoking_detector.utils.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')

#
# Import Utils from tensorflow object_detection directory
#
from smoking_detector.utils.utils import label_map_util
from smoking_detector.utils.utils import visualization_utils as vis_util

# utils -> ops as utils_ops
#   shape_utils
#   static_shape
#   core -> standard_fields as fields
# utils -> label_map_util
#   google.protobuf import text_format
#   protos import string_int_label_map_pb2
# utils -> visualization_utils as vis_util
#   core -> standard_fields as fields
#   shape_utils
#

# only runs inside api directory.  need to fix pathing.
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
#PATH_TO_FROZEN_GRAPH = '../smoking_detector/weights/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb'
# to make sure pipenv run python api/mockup.py runs.
PATH_TO_FROZEN_GRAPH = 'smoking_detector/weights/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb'
print(PATH_TO_FROZEN_GRAPH)
# List of the strings that is used to add correct label for each box.
#PATH_TO_LABELS = os.path.join('../smoking_detector/dataset/', 'mscoco_label_map.pbtxt')
PATH_TO_LABELS = os.path.join('smoking_detector/dataset/', 'mscoco_label_map.pbtxt')

print(PATH_TO_LABELS)

#
# Download model
#
"""print(pathlib.Path(MODEL_FILE))
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
"""
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
#category_index = label_map_util.create_category_index_from_labelmap('../smoking_detector/datasets/mscoco_label_map.pbtxt', use_display_name=True)
category_index = label_map_util.create_category_index_from_labelmap('smoking_detector/datasets/mscoco_label_map.pbtxt', use_display_name=True)

#
# Load Videos
#
url = "https://youtu.be/dQw4w9WgXcQ"
#url = "https://www.youtube.com/watch?v=UB4GdVs1UCM"
videoPafy = pafy.new(url)
best = videoPafy.getbest(preftype="mp4")

cap = cv2.VideoCapture()
cap.open(best.url)

count = 10
count_key = 'person'
count_image_lists = []

# Processing the whole video within one tf.Session() instead of creating
# a tf.Session() for every image in the video.
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        for i in range(count):
        #while True:
            ret, image_np = cap.read()
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

            # go through the boxes
            print(boxes.shape)
            foo = np.squeeze(classes).astype(np.int32)
            for j in range(boxes.shape[0]):
                if foo[j] in category_index.keys():
                    if ((category_index[foo[j]]['name'] == count_key) and
                        (int(np.squeeze(scores)[j]*100 >= 60))):
                        print("AFDADSFDSFS")
                        print(category_index[foo[j]]['name'], int(np.squeeze(scores)[j]*100))

                        # Visualization of the results of a detection.
                        vis_util.visualize_boxes_and_labels_on_image_array(
                            image_np,
                            np.squeeze(boxes),
                            np.squeeze(classes).astype(np.int32),
                            np.squeeze(scores),
                            category_index,
                            use_normalized_coordinates=True,
                            line_thickness=8)
                        print(str(i)+".png")
                        cv2.imwrite(str(i)+'.png', image_np)#, cv2.resize(image_np, (800,600)))

            cv2.imshow('object detection', image_np)#, cv2.resize(image_np, (800,600)))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
            #if i = 10:
            #    cv2.destroyAllWindows()
            #    break
        cv2.destroyAllWindows()
print("{0} images of {1} found. Processing stopped.".format(count, count_key))
        #break
