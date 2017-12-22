from __future__ import print_function
import model as modellib
from config import Config
import os

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


class InferenceConfig(VolumesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
inference_config = InferenceConfig()

ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
model = modellib.MaskRCNN(mode="inference", config=inference_config,model_dir=MODEL_DIR)
model.load_weights("logs/shapes20171220T0949/mask_rcnn_shapes_0028.h5",by_name=True)
original_image, image_meta, gt_bbox = modellib.load_image_gt(None,inference_config,-1, use_mini_mask=False)
results = model.detect([original_image], verbose=1)
from conversions.validateNNOutput import validateNNOutput, visualizeNNOutput

#it only prints the first output
#validateNNOutput(original_image,results[0]['rois'], gt_bbox, sleep_time=1000000000)
visualizeNNOutput(results[0]['rois'], gt_bbox,sleep_time=1000000000)