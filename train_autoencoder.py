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
    IMAGES_PER_GPU = 3
    NAME = "AUTOENCODER"

config = VolumesConfig()
print (config)

# from tensorflow.python import debug as tf_debug
# sess = K.get_session()
# sess = tf_debug.LocalCLIDebugWrapperSession(sess)
# K.set_session(sess)
#Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)
#model.load_weights("logs/autoencoder20180205T1959/mask_rcnn_autoencoder_0002.h5",by_name=True)
model.train(None,None,
            learning_rate=config.LEARNING_RATE / 10,
            epochs=100,
            layers="all")

