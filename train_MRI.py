from __future__ import print_function
import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

from config import Config
import utils
import model as modellib
import visualize
from model import log
import keras.backend as K

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to COCO trained weights
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

class VolumesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 organ

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 200

    # use small validation steps since the epoch is small
    VALIDATION_STPES = 5

config = VolumesConfig()
print (config)

# from nifti import *
# data_mri =  NiftiImage('conversions/MRI_orig_padded0_input_maskRCNN.nii').data
# import pandas as pd
# df=pd.read_csv('out.csv', sep=',')
# data = df.as_matrix()
# boxes = np.zeros((1,7))
# boxes[0,0]= int(data[0,2])
# boxes[0,1]= int(data[0,3])
# boxes[0,2]= int(data[0,4])
# boxes[0,3]= int(data[0,5])
# boxes[0,4]= int(data[0,6])
# boxes[0,5]= int(data[0,7])
# boxes[0,6] = 1

# from tensorflow.python import debug as tf_debug
# sess = K.get_session()
# sess = tf_debug.LocalCLIDebugWrapperSession(sess)
# K.set_session(sess)
# Create model in training mode
# model = modellib.MaskRCNN(mode="training", config=config,
#                           model_dir=MODEL_DIR)
# model.load_weights("logs/shapes20171220T0949/mask_rcnn_shapes_0023.h5")
# model.train(None,None,
#             learning_rate=config.LEARNING_RATE / 10,
#             epochs=100,
#             layers="all")


class InferenceConfig(VolumesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
inference_config = InferenceConfig()

model = modellib.MaskRCNN(mode="inference", config=inference_config,model_dir=MODEL_DIR)
model.load_weights("logs/shapes20171220T0949/mask_rcnn_shapes_0023.h5",by_name=True)
original_image, image_meta, gt_bbox = modellib.load_image_gt(None,inference_config,-1, use_mini_mask=False)
results = model.detect([original_image], verbose=1)
from conversions.validateNNOutput import validateNNOutput, visualizeNNOutput

#it only prints the first output
#validateNNOutput(original_image,results[0]['rois'], gt_bbox, sleep_time=1000000000)
visualizeNNOutput(results[0]['rois'], gt_bbox,sleep_time=1000000000)

print ("ok")

# batch_size = 1
# batch_image_meta = np.zeros((batch_size,)+image_meta.shape, dtype=image_meta.dtype)
# batch_rpn_match = np.zeros([batch_size, anchors.shape[0], 1], dtype=rpn_match.dtype)
# batch_rpn_bbox = np.zeros([batch_size, config.RPN_TRAIN_ANCHORS_PER_IMAGE, 6], dtype=rpn_bbox.dtype)
# batch_images = np.zeros((batch_size,)+image.shape, dtype=np.float32)
# batch_gt_boxes = np.zeros((batch_size, config.MAX_GT_INSTANCES, 7), dtype=np.int32)
# batch_images[0] = original_image
