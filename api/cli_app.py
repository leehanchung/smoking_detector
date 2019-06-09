# Video processing libraries
# Must compile and install opencv-python, won't work if pip install opencv-python
print("Loading libraries...")
import numpy as np
import tensorflow as tf
import click

import cv2, pafy, youtube_dl
from PIL import Image

from distutils.version import StrictVersion
if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')

# Import Utils from tensorflow object_detection directory
from smoking_detector.utils.utils import ops as utils_ops
from smoking_detector.utils.utils import label_map_util
from smoking_detector.utils.utils import visualization_utils as vis_util

print("Finishing loading libraries...")
PATH_TO_FROZEN_GRAPH = 'smoking_detector/weights/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb'
PATH_TO_LABELS = 'smoking_detector/datasets/mscoco_label_map.pbtxt'


# INPUT: Tensorflow graph and frozen model
# OUTPUT: Tensorflow graph with loaded weight from frozen label
def _load_model(graph, model):
    with graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(model, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    return graph


# INPUT: catogory index label map
# OUTPUT: parsed category index label map
def _load_label(label_map):
    return label_map_util.create_category_index_from_labelmap(label_map, use_display_name=True)


# INPUT: image in PIL format
# OUTPUT: image in np format.
def _load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


def image_detection(image_file):
    """
    image = Image.open(image_file)
    image.show()
    image_np = _load_image_into_numpy_array(image)
    """
    #print(image_file)
    image_np = cv2.imread(image_file, cv2.IMREAD_COLOR)
    #print(image_np)
    cv2.imshow('before', image_np)
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

            # Transforming image_np back to PIL format for display
            #detected_image = Image.fromarray(image_np, 'RGB')
            #detected_image.show()
            results = ""
            for i in range(3):
                ##class_name = category_index[classes[i]]['name']
                #results = results + category_index[np.squeeze(classes).astype(np.int32)[i]]['name'] + str(np.squeeze(scores)[i])
                print("{}: {:.2%}".format(category_index[np.squeeze(classes).astype(np.int32)[i]]['name'], np.squeeze(scores)[i]))
            print(results)
            print(np.squeeze(classes)[:5], np.squeeze(scores)[:5])
            cv2.imshow('object detection', image_np)
            print("Displaying image... \nPress 'q' to exit...")
            if cv2.waitKey(0) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
            #break
            #cv2.imshow('detected_image', image_np)
    return None


def video_detection():
    url = "https://youtu.be/dQw4w9WgXcQ"
    videoPafy = pafy.new(url)
    best = videoPafy.getbest(preftype="mp4")

    cap = cv2.VideoCapture()
    cap.open(best.url)

    count = 10
    count_key = 'person'
    count_image_lists = []

    # Processing the whole video within one tf.Session() instead of creating
    # a tf.Session() for every image in the video.
    category_index = _load_label(PATH_TO_LABELS)
    detection_graph = _load_model(tf.Graph(), PATH_TO_FROZEN_GRAPH)
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            for i in range(count):
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
            cv2.destroyAllWindows()
    print("{0} images of {1} found. Processing stopped.".format(count, count_key))
    return None

"""@click.command()
@click.option('--image', type=str, default='smoking_detector/datasets/image1.jpg')
@click.option('--video', type=str, default='https://youtu.be/dQw4w9WgXcQ')
"""
def main():#image, video):
    # some click problems.
    #if video is None:
    """
    print("Detecting smoking videos...")
    video_detection()
    else:
    """
    print("Detecting smoking images...")
    image_detection('smoking_detector/datasets/image1.jpg')

if __name__ == "__main__":
    main()
