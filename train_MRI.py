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

class VolumesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"

config = VolumesConfig()
print (config)

# from tensorflow.python import debug as tf_debug
# sess = K.get_session()
# sess = tf_debug.LocalCLIDebugWrapperSession(sess)
# K.set_session(sess)
# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)
model.load_weights("savedModels/mask_rcnn_shapes_0033-14.02.2018.h5",by_name=True) ##### MRI trained model
#model.load_pretrained_weights("savedModels/mask_rcnn_autoencoder_0017-02.14.2018.h",by_name=True) ##### Autoencoder model
model.train(None,None,
            learning_rate=config.LEARNING_RATE / 10,
            epochs=100,
            layers="heads")

