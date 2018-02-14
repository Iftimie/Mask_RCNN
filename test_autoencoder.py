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

from config_autoencoder import Config
import utils
import autoencoder_model as modellib
import visualize
from model import log
import keras.backend as K

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

class VolumesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    IMAGES_PER_GPU = 1
    NAME = "AUTOENCODER"

config = VolumesConfig()
class InferenceConfig(VolumesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
inference_config = InferenceConfig()

# from tensorflow.python import debug as tf_debug
# sess = K.get_session()
# sess = tf_debug.LocalCLIDebugWrapperSession(sess)
# K.set_session(sess)
#Create model in training mode
model = modellib.MaskRCNN(mode="inference", config=config, model_dir=MODEL_DIR)
model.load_weights("savedModels/mask_rcnn_autoencoder_0017-02.14.2018.h",by_name=True)
#model.load_weights("logs/autoencoder20180205T1959/mask_rcnn_autoencoder_0008.h5",by_name=True)

cv2.namedWindow('reconstructed image',cv2.WINDOW_NORMAL)
cv2.namedWindow('original image',cv2.WINDOW_NORMAL)
from nifti import *
for image_id in range(1,19):
    original_image= modellib.load_image_gt(None,inference_config,image_id=image_id, use_mini_mask=False)
    results = model.detect([original_image], verbose=1, config=inference_config)
    RMI = results[0]['image']
    import cv2
    for x in range(128):
        #the output here is after RELU so it is not less than 0, but it is possible to be more than 1. Should clamp the values before multiplying
        RMI = np.clip(RMI,0.0,1.0)
        slice = np.array(RMI[:,:,x] * 255,dtype=np.uint8)
        slice = cv2.resize(slice,(0,0), fx=2, fy=2)
        cv2.imshow("reconstructed image",slice)
        slice = np.array(original_image[:,:,x],dtype=np.uint8)
        slice = cv2.resize(slice, (0,0), fx=2, fy=2)
        cv2.imshow("original image",slice)
        cv2.waitKey(100)
        if image_id == 1 and x==1:
            cv2.waitKey(100)

    import scipy.ndimage as ndimage
    RMI = RMI[...,0] * 255
    resized_data = ndimage.zoom(RMI,(2.0)) #ndimage.zoom(data,0.5)
    print (resized_data.shape)
    resized_data = resized_data.astype(np.uint8)
    nim = NiftiImage(resized_data)
    nim.save("../autoencoder_output.nii")
