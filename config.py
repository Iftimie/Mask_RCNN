# coding=utf-8
"""
Mask R-CNN
Base Configurations class.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import math
import numpy as np


# Base Configuration Class
# Don't use this class directly. Instead, sub-class it and override
# the configurations you need to change.

class Config(object):
    """Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    """
    # Name the configurations. For example, 'COCO', 'Experiment 3', ...etc.
    # Useful if your code needs to do things differently depending on which
    # experiment is running.
    NAME = None  # Override in sub-classes

    # NUMBER OF GPUs to use. For CPU training, use 1
    GPU_COUNT = 1

    # Number of images to train with on each GPU. A 12GB GPU can typically
    # handle 2 images of 1024x1024px.
    # Adjust based on your GPU memory and image sizes. Use the highest
    # number that your GPU can handle for best performance.
    IMAGES_PER_GPU = 5

    # Number of training steps per epoch
    # This doesn't need to match the size of the training set. Tensorboard
    # updates are saved at the end of each epoch, so setting this to a
    # smaller number means getting more frequent TensorBoard updates.
    # Validation stats are also calculated at each epoch end and they
    # might take a while, so don't set this too small to avoid spending
    # a lot of time on validation stats.
    STEPS_PER_EPOCH = 200

    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    VALIDATION_STPES = 5

    # The strides of each layer of the FPN Pyramid. These values
    # are based on a Resnet101 backbone.
    BACKBONE_STRIDES = [4, 8, 16, 32] #there are 5 values. probably each for each stage C1, C2, C3, C4. before it was also for C5

    # Number of classification classes (including background)
    NUM_CLASSES = 1 + 6 # Override in sub-classes

    ###### RPN ######
    # Length of square anchor side in pixels
    ## Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (6, 12, 24, 48)  # anchor side in pixels #  the number of scales should be equal to BACKBONE_STRIDES
    #this is clever because you compute the anchors for specific layers. the low resolution layers contain information about big objects
    #high resolution layers contain information about smaller objects
    #so there are attached achros to specific layers. this is better because if we attach small scale on low resolution, the
    # low resolution layer is too compresed to contain fine detail about small objects
    #low resolution layers work best with big scale anchors

    # Ratios of anchors at each cell (width/height)
    # A value of 1 represents a square anchor, and 0.5 is a wide anchor
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]
    NUM_ANCHORS_PER_LOCATION = 4 # in 3D there are 4 boxes per location not 3 as before. with the coresponding ratios
    # there is generated 1 anchor for every two every two voxels in the maps
    # maps have the following sizes: 32^3, 16^3, 8^3, 4^3, 2^3 but our number of anchors is (16^3+8^3+4^3+2^3+1^3)×4 =18724
    #before for an image of 128 by 128 with 3 aspect ratios: 1023 anchors. what about 640*640
    #one problem that i had was the fact that I was generating fewer proposals than anchors because in build_rpn_model(num_anchors = 3) instead of 4
    #on the GPU there was no problem, explanation here:https://github.com/tensorflow/tensorflow/issues/15091 comment by ebrevdo
    #on the CPU however the indices are validated, and the problem was recognized
    #because the laptop did not recognized the GPU and for a while showed this problem I was able to find this bug


    # Anchor stride
    # If 1 then anchors are created for each cell in the backbone feature map.
    # If 2, then anchors are created for every other cell, and so on.
    RPN_ANCHOR_STRIDE = 1

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 512

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 10000
    POST_NMS_ROIS_INFERENCE = 3000

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Input image resing
    # Images are resized such that the smallest side is >= IMAGE_MIN_DIM and
    # the longest side is <= IMAGE_MAX_DIM. In case both conditions can't
    # be satisfied together the IMAGE_MAX_DIM is enforced.
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128
    # If True, pad images with zeros such that they're (max_dim by max_dim)
    IMAGE_PADDING = True  # currently, the False option is not supported

    # Image mean (RGB)
    MEAN_PIXEL = np.array([127.5])

    # Number of ROIs per image to feed to classifier/mask heads
    TRAIN_ROIS_PER_IMAGE = 512  # TODO: paper uses 512

    # Percent of positive ROIs used to train classifier/mask heads
    ROI_POSITIVE_RATIO = 0.33

    # Pooled ROIs
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14
    MASK_SHAPE = [28, 28]

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 6

    DEV = 0.1 # DEV = 0.1 for x y z and 0.2 for h w d
    DEV2 =0.2
    # Bounding box refinement standard deviation for RPN and final detections.
    RPN_BBOX_STD_DEV = np.array([DEV, DEV, DEV, DEV2, DEV2, DEV2])
    BBOX_STD_DEV = np.array([DEV, DEV, DEV, DEV2, DEV2, DEV2])

    # Max number of final detections
    DETECTION_MAX_INSTANCES = 100

    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.5 #original 0.7

    RPN_ROIS_IOU_GT_BOX_POSITIVE_TRESH = 0.3 #original 0.5 #this is between proposed rois and ground-truth images

    # You can reduce this during training to generate more propsals
    NMS_TRESHOLD_ANCHORS_AFTER_APPLY_DELTAS = 0.7 #original: 0.7 found as RPN_NMS_THRESHOLD

    # Non-maximum suppression threshold for detection/ inference
    DETECTION_NMS_THRESHOLD = 0.3 #original 0.3

    ANCHOR_IOU_POS_TRESH = 0.3 #original 0.7 and 0.3
    ANCHOR_IOU_NEG_TRESH = 0.2

    # Learning rate and momentum
    # The paper uses lr=0.02, but we found that to cause weights to explode often
    LEARNING_RATE = 0.0006 #this
    LEARNING_MOMENTUM = 0.9

    # Weight decay regularization
    WEIGHT_DECAY = 0.0001

    # Use RPN ROIs or externally generated ROIs for training
    # Keep this True for most situations. Set to False if you want to train
    # the head branches on ROI generated by code rather than the ROIs from
    # the RPN. For example, to debug the classifier head without having to
    # train the RPN.
    USE_RPN_ROIS = True

    def __init__(self):
        """Set values of computed attributes."""
        # Effective batch size
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT

        # Input image size
        self.IMAGE_SHAPE = np.array([self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM, 1])

        # Compute backbone size from input image size
        self.BACKBONE_SHAPES = np.array(
            [[int(math.ceil(self.IMAGE_SHAPE[0] / stride)),
              int(math.ceil(self.IMAGE_SHAPE[1] / stride)),
              int(math.ceil(self.IMAGE_SHAPE[2] / stride))]
             for stride in self.BACKBONE_STRIDES])

    def __print__(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
