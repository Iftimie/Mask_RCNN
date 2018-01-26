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





class InferenceConfig(VolumesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
inference_config = InferenceConfig()

ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
model = modellib.MaskRCNN(mode="inference", config=inference_config,model_dir=MODEL_DIR)
model.load_weights("logs/shapes20180126T1615/mask_rcnn_shapes_0039.h5",by_name=True)


original_image, image_meta, gt_bbox = modellib.load_image_gt(None,inference_config,image_id=1, use_mini_mask=False)
results = model.detect([original_image], verbose=1, config=inference_config)
print (results)
from conversions.validateNNOutput import validateNNOutput, visualizeNNOutput

#it only prints the first output
#validateNNOutput(original_image,results[0]['rois'], gt_bbox, sleep_time=10000)
visualizeNNOutput(results[0]['rois'], gt_bbox,sleep_time=6)