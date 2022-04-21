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

from flask import Flask, request
from flask import jsonify
from posix import POSIX_FADV_WILLNEED
import json



PATH_TO_MODEL_DIR = '/home/sanjeev/projects/mtp-1/TFOD_FRCNN(1)/TFOD/workspace/train/exported-models/frcnn'

# PROVIDE PATH TO LABEL MAP
PATH_TO_LABELS = '/home/sanjeev/projects/mtp-1/TFOD_FRCNN(1)/TFOD/workspace/train/annotations/label_map.pbtxt'

# PROVIDE THE MINIMUM CONFIDENCE THRESHOLD
min_score_thresh = float(0.60)

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

color_map = {1:8, 2:98}

def infer(IMAGE_PATH, pg_no, content):
  start = time.time()
  print('Running inference for {}... '.format(IMAGE_PATH))
  image = cv2.imread(IMAGE_PATH)
  input_tensor = tf.convert_to_tensor(image)
  input_tensor = input_tensor[tf.newaxis, ...]
  detections = detect_fn(input_tensor)
  end = time.time()

  print("Inference time : ", end-start, "seconds")

  start = time.time()
  num_detections = int(detections.pop('num_detections'))
  detections = {key: value[0, :num_detections].numpy()
                for key, value in detections.items()}
  detections['num_detections'] = num_detections
  detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
  image_with_detections = image.copy()

  arr = []

  for num in detections['detection_classes']:
    arr.append(color_map[num])

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
        min_score_thresh=0.8,
        agnostic_mode=False,
        line_thickness=12)

  print('Done') 
  end = time.time()
  print("Visualise time : ", end-start, "seconds")
  cv2.imwrite(os.path.join(IMAGE_PATH), image_with_detections)
  print("Write time : ", time.time()-end, "seconds")

  #save the detected boxes to content
  content['P'+pg_no] = filtered_boxes




app = Flask(__name__)
@app.route('/', methods=(["POST"]))
def hello():
    if request.method == "POST":
        # print(request.data)

        #content will contain a map of page -> detected bounding boxes
        content ={}
        #save the received pdf file on server
        try:
          os.mkdir("pdf")
        except:
          pass
        f = open("pdf/new_file.pdf", "wb")
        f.write(request.data)
        f.close()

        #save the images from pdf
        try:
          os.mkdir("images")
        except:
          pass
        os.system("python3 create_images_from_pdf.py pdf images")

        #infer for rach image
        for image in os.listdir("images"):
          image_path = os.path.join("images", image)
          infer(image_path, image.split('.')[0], content)
        # os.system("rm -rf darknet/aidetect && mkdir darknet/aidetect/")
        # cmdToRun = "cp new_file.png darknet/"
        # print("Copying new_file.png to darknet/")
        # os.system(cmdToRun)
        # cmdToRun = ""
        # cmdToRun += "cd darknet/ && rm -rf outputs/*.txt &&"
        # cmdToRun += "./darknet detector test data/multiple_images.data cfg/math.cfg backup/math_final.weights new_file.png -thresh 0.3 -save_labels -dont_show"
        # os.system(cmdToRun)
        # os.system("mv darknet/new_file.txt darknet/aidetect/0.txt")

        # os.system("python3 convertTojson.py darknet/aidetect/")

    # f = open("darknet/aidetect/0.json", "rb")
    # content = f.read()
    # # print(content)

    # f = open("new_file.png", 'rb')
    # content = f.read()

    print(content)
    resp = jsonify(success=True)
    resp.data = json.dumps(content)
    resp.status_code = 200
    
    os.system("rm -rf images")
    os.system("rm -rf pdf")

    return resp

if __name__ == "__main__":
    app.run(host ='0.0.0.0', port = 5001,debug = True)
