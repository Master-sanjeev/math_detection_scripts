from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from posix import POSIX_FADV_WILLNEED
import utils
import sys
import os
from os import listdir
from os.path import isfile, join
from PyPDF2 import pdf
import warnings
warnings.filterwarnings('ignore')
import abc
import collections
# Set headless-friendly backend.
import matplotlib; matplotlib.use('Agg')  # pylint: disable=multiple-statements
import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
import numpy as np
import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import six
from six.moves import range
from six.moves import zip
import tensorflow.compat.v1 as tf

from object_detection.core import keypoint_ops
from object_detection.core import standard_fields as fields
from object_detection.utils import shape_utils

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import pathlib
import tensorflow as tf
import cv2
import argparse

import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt



PATH_TO_MODEL_DIR = '/home/sanjeev/projects/mtp-1/TFOD_FRCNN(1)/TFOD/workspace/train/exported-models/frcnn'

# PROVIDE PATH TO LABEL MAP
PATH_TO_LABELS = '/home/sanjeev/projects/mtp-1/TFOD_FRCNN(1)/TFOD/workspace/train/annotations/label_map.pbtxt'

# PROVIDE THE MINIMUM CONFIDENCE THRESHOLD
MIN_CONF_THRESH = float(0.60)

# LOAD THE MODEL

PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"

print('Loading model...', end='')
start_time = time.time()

# LOAD SAVED MODEL AND BUILD DETECTION FUNCTION
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

# LOAD LABEL MAP DATA FOR PLOTTING

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)


# In[26]:


map = {1:8, 2:98}
# raw_detections = None
# IMAGES_PATH = "/home/sanjeev/projects/mtp-1/TFOD_FRCNN(1)/TFOD/images" 
IMAGES_PATH = '/'+sys.argv[1].strip('/')

output_path = os.path.join("/".join(IMAGES_PATH.split('/')[:-1]), 'output')
if not os.path.exists(output_path) :
    os.mkdir(os.path.join(output_path))
    
json_path = os.path.join("/".join(IMAGES_PATH.split('/')[:-1]), 'aidetect')
if not os.path.exists(json_path) :
    os.mkdir(os.path.join(json_path))
  
for IMAGE_PATH in os.listdir(IMAGES_PATH):
  pg_no = IMAGE_PATH.split('.')[0]
  start = time.time()
  img = IMAGE_PATH
  IMAGE_PATH = os.path.join(IMAGES_PATH, IMAGE_PATH)
  print('Running inference for {}... '.format(IMAGE_PATH))

  image = cv2.imread(IMAGE_PATH)
#   image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#   image_expanded = np.expand_dims(image_rgb, axis=0)

  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis, ...]

  # input_tensor = np.expand_dims(image_np, 0)
  detections = detect_fn(input_tensor)
#   raw_detections = detections.copy()
    
  end = time.time()

  print("Inference time : ", end-start, "seconds")
  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.

  start = time.time()
  num_detections = int(detections.pop('num_detections'))
  # print("num detections ",num_detections)
  detections = {key: value[0, :num_detections].numpy()
                for key, value in detections.items()}
  detections['num_detections'] = num_detections
  # print("printing detections : "+str(detections))
  # detection_classes should be ints.
  detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
  # print(detections)
  image_with_detections = image.copy()
  arr = []
  for num in detections['detection_classes']:
    arr.append(map[num])
  # SET MIN_SCORE_THRESH BASED ON YOU MINIMUM THRESHOLD FOR DETECTIONS
  filtered_boxes = utils.visualize_boxes_and_labels_on_image_array(
        pg_no,
        image_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        track_ids=np.asarray(arr),
        max_boxes_to_draw=None,
        min_score_thresh=0.60,
        agnostic_mode=False,
        line_thickness=12)

  print('Done')
  end = time.time()
  print("Visualise time : ", end-start, "seconds")
  # DISPLAYS OUTPUT IMAGE
#   cv2_imshow(image_with_detections)
#   cv2.imshow("detection", image_with_detections)
#   cv2.waitKey(0) 
  
  #closing all open windows 
#   cv2.destroyAllWindows() 
  cv2.imwrite(os.path.join(output_path, img), image_with_detections)
  print("Write time : ", time.time()-end, "seconds")


  f = open('{}.txt'.format(os.path.join(json_path, pg_no)), 'w')
  for box in filtered_boxes : 
    f.write(str(box[0])+' '+str(box[1])+' '+str(box[2])+' '+str(box[3])+' '+str(box[4])+'\n')

  f.close()
