"""
Mask R-CNN
The main Mask R-CNN model implemenetation.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import os
import datetime
import re
import logging
from collections import OrderedDict
import numpy as np
import scipy.misc
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM
import keras.losses as KLO


import utils

# Requires TensorFlow 1.3+ and Keras 2.0.8+.
from distutils.version import LooseVersion
assert LooseVersion(tf.__version__) >= LooseVersion("1.3")
assert LooseVersion(keras.__version__) >= LooseVersion('2.0.8')


def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  min: {:10.5f}  max: {:10.5f}".format(
            str(array.shape),
            array.min() if array.size else "",
            array.max() if array.size else ""))
    print(text)

############################################################
#  Utility Functions
############################################################


class BatchNorm(KL.BatchNormalization):
    """Batch Normalization class. Subclasses the Keras BN class and
    hardcodes training=False so the BN layer doesn't update
    during training.

    Batch normalization has a negative effect on training if batches are small
    so we disable it here.
    """
    def call(self, inputs, training=None):
        return super(self.__class__, self).call(inputs, training=False)


############################################################
#  Resnet Graph
############################################################

# Code adopted from:
# https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py

def identity_block(input_tensor, kernel_size, filters, stage, block,
                   use_bias=True):
    """The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    """
    nb_filter2 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'

    x = KL.Conv3D(nb_filter2, (kernel_size, kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(input_tensor)
    x = KL.Activation('relu')(x)

    return x

#example call conv_block(x, 3, [2, 2, 8], stage=2, block='a', strides=(1, 1, 1))
def conv_block(input_tensor, kernel_size, filters, stage, block,
               strides=(2, 2, 2), use_bias=True):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """
    nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'

    x = KL.Conv3D(nb_filter2, (kernel_size, kernel_size, kernel_size), padding='same', strides =strides,
                  name=conv_name_base + '2b', use_bias=use_bias)(input_tensor)
    x = KL.Activation('relu')(x)

    x = KL.Conv3D(nb_filter3, (kernel_size, kernel_size, kernel_size), padding='same', name=conv_name_base + '2c', use_bias=use_bias)(x)

    x = KL.Activation('relu', name='res'+str(stage)+block+'_out')(x)
    return x


def resnet_graph(input_image, architecture, stage5=False):
    assert architecture in ["resnet50", "resnet101"]
    # Stage 1
    x = KL.ZeroPadding3D((3, 3, 3))(input_image)
    x = KL.Conv3D(8, (7, 7, 7), strides=(2, 2, 2), name='conv1', use_bias=True)(x)
    C1 = x = KL.Activation('relu')(x)

    # Stage 2
    C2 = x = conv_block(x, 3, [8, 16], stage=2, block='a')
    # x = identity_block(x, 3, [16], stage=2, block='b')
    # C2 = x = identity_block(x, 3, [2, 2, 8], stage=2, block='c')
    # Stage 3
    C3 = x = conv_block(x, 3, [16, 32], stage=3, block='a')
    # x = identity_block(x, 3, [4, 4, 16], stage=3, block='b')
    # x = identity_block(x, 3, [4, 4, 16], stage=3, block='c')
    # C3 = x = identity_block(x, 3, [4, 4, 16], stage=3, block='d')
    # Stage 4
    x = conv_block(x, 3, [32, 64], stage=4, block='a')
    # block_count = {"resnet50": 5, "resnet101": 22}[architecture]
    # for i in range(block_count):
    #     x = identity_block(x, 3, [8, 8, 32], stage=4, block=chr(98+i))
    C4 = x
    # Stage 5
    if stage5:
        C5 = x = conv_block(x, 3, [64, 128], stage=5, block='a')
        #x = identity_block(x, 3, [16, 16, 64], stage=5, block='b')
        #C5 = x = identity_block(x, 3, [16, 16, 64], stage=5, block='c')
    else:
        C5 = None
    return [C1, C2, C3, C4, C5]

def reverse_resnet_graph(layer, architecture, stage5=False):
    # # Stage 5
    if stage5:
        layer = KL.UpSampling3D((2,2,2))(layer)
        layer = conv_block(layer, 3,[64, 64], stage=5, block='arev', strides=(1,1,1))
    # Stage 4
    #block_count = {"resnet50": 5, "resnet101": 22}[architecture]
    layer = KL.UpSampling3D((2,2,2))(layer)
    # layer = KL.Conv3D(32, (3,3,3), strides=(1, 1, 1), use_bias=True, padding='same')(layer)
    # layer = KL.Activation('relu')(layer)
    # for i in range(block_count):
    #     layer = reverse_identity_block(layer, 3, [8, 8, 32], stage=4, block=chr(98 + block_count -1 - i))
    layer = conv_block(layer, 3, [32, 32], stage=4, block='arev', strides=(1,1,1))
    # Stage 3
    layer = KL.UpSampling3D((2,2,2))(layer)
    # layer = KL.Conv3D(16, (3,3,3), strides=(1, 1, 1), use_bias=True, padding='same')(layer)
    # layer = KL.Activation('relu')(layer)
    # layer = reverse_conv_block(layer, 3, [4, 4, 16], stage=3, block='d')
    # layer = reverse_identity_block(layer, 3, [4, 4, 16], stage=3, block='c')
    # layer = reverse_identity_block(layer, 3, [4, 4, 16], stage=3, block='b')
    layer = conv_block(layer,3, [16, 16], stage=3, block='arev', strides=(1,1,1))
    # Stage 2
    layer = KL.UpSampling3D((2,2,2))(layer)
    # layer = KL.Conv3D(8, (3,3,3), strides=(1, 1, 1), use_bias=True, padding='same')(layer)
    # layer = KL.Activation('relu')(layer)
    # layer = reverse_identity_block(layer, 3, [2, 2, 8], stage=2, block='c')
    # layer = reverse_identity_block(layer, 3, [2, 2, 8], stage=2, block='b')
    layer = conv_block(layer, 3, [8, 8], stage=2, block='arev', strides=(1,1,1))
    # Stage 1
    layer = KL.UpSampling3D((2,2,2))(layer)
    layer = KL.Conv3D(1, (7, 7, 7), strides=(1, 1, 1), name='conv_last', use_bias=True, padding='same')(layer)
    layer = KL.Activation('relu')(layer)
    return layer


############################################################
#  Data Generator
############################################################
class_dictionary={0:1, 1:2, 2:3, 9:4, 10:5, 12:6}
def load_image_gt(dataset, config, image_id, augment=False,use_mini_mask=False):
    """Load and return ground truth data for an image (image, mask, bounding boxes).

    augment: If true, apply random image augmentation. Currently, only
        horizontal flipping is offered.
    use_mini_mask: If False, returns full-size masks that are the same height
        and width as the original image. These can be big, for example
        1024x1024x100 (for 100 instances). Mini masks are smaller, typically,
        224x224 and are generated by extracting the bounding box of the
        object and resizing it to MINI_MASK_SHAPE.

    Returns:
    image: [height, width, depth, 1]
    shape: the original shape of the image before resizing and cropping.
    bbox: [instance_count, (y1, x1, z1, y2, x2, z2, class_id)]
    mask: [height, width, instance_count]. The height and width are those
        of the image unless use_mini_mask is True, in which case they are
        defined in MINI_MASK_SHAPE.
    """
    # Load image and mask
    #image = dataset.load_image(image_id)
    #from nifti import NiftiImage
    import nibabel as nib
    data_mri = nib.load('input_MaskRCNN_128/MRI_'+str(image_id)+'.nii').get_data()
    #data_mri =  NiftiImage('input_MaskRCNN_128/MRI_'+str(image_id)+'.nii').data
    image = data_mri[:,:,:,np.newaxis]


    return image


def data_generator(dataset, config, shuffle=True, augment=True, random_rois=0,
                   batch_size=1, detection_targets=False):
    """A generator that returns volumes and corresponding target class ids,
    bounding box deltas, and masks.

    dataset: The Dataset object to pick data from
    config: The model config object
    shuffle: If True, shuffles the samples before every epoch
    augment: If True, applies image augmentation to images (currently only
             horizontal flips are supported)
    random_rois: If > 0 then generate proposals to be used to train the
                 network classifier and mask heads. Useful if training
                 the Mask RCNN part without the RPN.
    batch_size: How many images to return in each call
    detection_targets: If True, generate detection targets (class IDs, bbox
        deltas, and masks). Typically for debugging or visualizations because
        in trainig detection targets are generated by DetectionTargetLayer.

    Returns a Python generator. Upon calling next() on it, the
    generator returns two lists, inputs and outputs. The containtes
    of the lists differs depending on the received arguments:
    inputs list:
    - images: [batch, H, W, D, C]
    - image_meta: [batch, size of image meta]
    - rpn_match: [batch, N] Integer (1=positive anchor, -1=negative, 0=neutral)
    - rpn_bbox: [batch, N, (dy, dx, dz, log(dh), log(dw), log(dd))] Anchor bbox deltas.
    - gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, z1, y2, x2, z2, class_id)]
    - gt_masks: [batch, height, width, MAX_GT_INSTANCES]. The height and width
                are those of the image unless use_mini_mask is True, in which
                case they are defined in MINI_MASK_SHAPE.

    outputs list: Usually empty in regular training. But if detection_targets
        is True then the outputs list contains target class_ids, bbox deltas,
        and masks.
    """
    b = 0  # batch item index
    image_index = 0
    image_ids = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]) # i only have 6 images
    error_count = 0

    # Keras requires a generator to run indefinately.
    while True:
        try:

            # Get GT bounding boxes and masks for image.
            image_id = image_ids[image_index]
            image_index = (image_index + 1) % 5

            image = load_image_gt(dataset, config, image_id, augment=augment, use_mini_mask=config.USE_MINI_MASK)

            # Init batch arrays
            if b == 0:
                batch_images = np.zeros((batch_size,)+image.shape, dtype=np.float32)

            # Add to batch
            batch_images[b] = mold_image(image.astype(np.float32), config)
            b += 1
            # Batch full?
            if b >= batch_size:
                inputs = [batch_images]
                outputs = []

                yield inputs, outputs

                # start a new batch
                b = 0
        except (GeneratorExit, KeyboardInterrupt):
            raise
        except:
            # Log it and skip the image
            logging.exception("Error processing image {}".format(image_id))
            error_count += 1
            if error_count > 5:
                raise



############################################################
#  MaskRCNN Class
############################################################

class MaskRCNN():
    """Encapsulates the Mask RCNN model functionality.

    The actual Keras model is in the keras_model property.
    """

    def __init__(self, mode, config, model_dir):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        assert mode in ['training', 'inference']
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()
        self.keras_model = self.build(mode=mode, config=config)

    def loss_function(self, labels,logits):
        loss = tf.losses.mean_squared_error(labels=labels,predictions=logits, weights=20.0)
        #loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,logits=logits)
        # loss = K.mean(loss)
        # loss = K.reshape(loss, [1, 1])
        return loss

    def build(self, mode, config):
        """Build Mask R-CNN architecture.
            input_shape: The shape of the input image.
            mode: Either "training" or "inference". The inputs and
                outputs of the model differ accordingly.
        """
        assert mode in ['training', 'inference']

        # Image size must be dividable by 2 multiple times
        h, w, d = config.IMAGE_SHAPE[:3]
        if h/2**6 != int(h/2**6) or w/2**6 != int(w/2**6) or d/2**6 != int(d/2**6):
            raise Exception("Volume size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")

        # Inputs
        input_image = KL.Input(shape=config.IMAGE_SHAPE.tolist(), name="input_image")


        # Build the shared convolutional layers.
        # Bottom-up Layers
        # Returns a list of the last layers of each stage, 5 in total.
        # Don't create the head (stage 5), so we pick the 4th item in the list.
        _, C2, C3, C4, C5 = resnet_graph(input_image, "resnet101", stage5=False)

        output_logits = reverse_resnet_graph(C4, "resnet50", stage5=False)
        output = output_logits
        #output = KL.Activation('sigmoid', name='rev_res')(output_logits)


        loss = KL.Lambda(lambda x: self.loss_function(*x), name="loss")([input_image, output_logits])

        # Top-down Layers


        if mode == "training":

            model = KM.Model([input_image], [ loss], name='mask_rcnn')
        else:

            model = KM.Model([input_image], [output], name='mask_rcnn')

        return model

    def load_weights(self, filepath, by_name=False, exclude=None):
        """Modified version of the correspoding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exlude: list of layer names to excluce
        """
        import h5py
        from keras.engine import topology

        if exclude:
            by_name = True

        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        f = h5py.File(filepath, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        keras_model = self.keras_model
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model") \
            else keras_model.layers

        # Exclude some layers
        if exclude:
            layers = filter(lambda l: l.name not in exclude, layers)

        if by_name:
            topology.load_weights_from_hdf5_group_by_name(f, layers)
        else:
            topology.load_weights_from_hdf5_group(f, layers)
        if hasattr(f, 'close'):
            f.close()

        # Update the log directory
        self.set_log_dir(filepath)

    def compile(self, learning_rate, momentum):
        """Gets the model ready for training. Adds losses, regularization, and
        metrics. Then calls the Keras compile() function.
        """
        # Optimizer object
        optimizer = keras.optimizers.Adam(lr=learning_rate)
        # Add Losses
        # First, clear previously set losses to avoid duplication
        self.keras_model._losses = []
        self.keras_model._per_input_losses = {}
        loss_names = ["loss"]
        for name in loss_names:
            layer = self.keras_model.get_layer(name)
            if layer.output in self.keras_model.losses:
                continue
            self.keras_model.add_loss(tf.reduce_mean(layer.output, keep_dims=True))

        # Add L2 Regularization
        reg_losses = [keras.regularizers.l2(self.config.WEIGHT_DECAY)(w)
                      for w in self.keras_model.trainable_weights]
        self.keras_model.add_loss(tf.add_n(reg_losses))

        # Compile
        self.keras_model.compile(optimizer=optimizer, loss=[None]*len(self.keras_model.outputs))

        # Add metrics
        for name in loss_names:
            if name in self.keras_model.metrics_names:
                continue
            layer = self.keras_model.get_layer(name)
            self.keras_model.metrics_names.append(name)
            self.keras_model.metrics_tensors.append(tf.reduce_mean(layer.output,
                                                                   keep_dims=True))

    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.

        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """
        # Set date and epoch counter as if starting a new model
        self.epoch = 0
        now = datetime.datetime.now()

        # If we have a model path with date and epochs use them
        if model_path:
            # Continue from we left of. Get epoch and date from the file name
            # A sample model path might look like:
            # /path/to/logs/coco20171029T2315/mask_rcnn_coco_0001.h5
            regex = r".*/\w+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/mask\_rcnn\_\w+(\d{4})\.h5"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                self.epoch = int(m.group(6)) + 1

        # Directory for training logs
        self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(
            self.config.NAME.lower(), now))

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "mask_rcnn_{}_*epoch*.h5".format(
            self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace("*epoch*", "{epoch:04d}")

    def set_trainable(self, layer_regex, keras_model=None, indent=0, verbose=1):
        """Sets model layers as trainable if their names match
        the given regular expression.
        """
        # Print message on the first call (but not on recursive calls)
        if verbose > 0 and keras_model is None:
            log("Selecting layers to train")

        keras_model = keras_model or self.keras_model

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model") \
            else keras_model.layers

        def fullmatch(regex, string, flags=0):
            """Emulate python-3.4 re.fullmatch()."""
            return re.match("(?:" + regex + r")\Z", string, flags=flags)
        for layer in layers:
            # Is the layer a model?
            if layer.__class__.__name__ == 'Model':
                print("In model: ", layer.name)
                self.set_trainable(layer_regex, keras_model=layer, indent=indent+4)
                continue

            if not layer.weights:
                continue
            # Is it trainable?
            trainable = bool(fullmatch(layer_regex, layer.name))
            # Update layer. If layer is a container, update inner layer.
            if layer.__class__.__name__ == 'TimeDistributed':
                layer.layer.trainable = trainable
            else:
                layer.trainable = trainable
            # Print trainble layer names
            if trainable and verbose > 0:
                log("{}{:20}   ({})".format(" " * indent, layer.name,
                                            layer.__class__.__name__))

    def train(self, train_dataset, val_dataset, learning_rate, epochs, layers):
        """Train the model.
        train_dataset, val_dataset: Training and validation Dataset objects.
        learning_rate: The learning rate to train with
        epochs: Number of training epochs. Note that previous training epochs
                are considered to be done alreay, so this actually determines
                the epochs to train in total rather than in this particaular
                call.
        layers: Allows selecting wich layers to train. It can be:
            - A regular expression to match layer names to train
            - One of these predefined values:
              heaads: The RPN, classifier and mask heads of the network
              all: All the layers
              3+: Train Resnet stage 3 and up
              4+: Train Resnet stage 4 and up
              5+: Train Resnet stage 5 and up
        """
        assert self.mode == "training", "Create model in training mode."

        # Pre-defined layer regular expressions
        layer_regex = {
            # all layers but the backbone
            "heads": r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # From Resnet stage 4 layers and up
            "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "5+": r"(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # All layers
            "all": ".*",
        }
        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        # Data generators
        train_generator = data_generator(train_dataset, self.config, shuffle=True,
                                         batch_size=self.config.BATCH_SIZE)
        val_generator = data_generator(val_dataset, self.config, shuffle=True,
                                       batch_size=self.config.BATCH_SIZE)

        # Callbacks
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                        histogram_freq=0, write_graph=True, write_images=False),
            keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                            verbose=0, save_weights_only=True)
        ]

        # Common parameters to pass to fit_generator()
        fit_kwargs = {
            "steps_per_epoch": self.config.STEPS_PER_EPOCH,
            "callbacks": callbacks,
            #"validation_data": next(val_generator),
            "validation_steps": self.config.VALIDATION_STPES,
            "max_queue_size": 100,
            "workers": max(self.config.BATCH_SIZE // 2, 2),
            "use_multiprocessing": True,
        }

        # Train
        log("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
        log("Checkpoint Path: {}".format(self.checkpoint_path))
        self.set_trainable(layers)
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)

        self.keras_model.fit_generator(
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            **fit_kwargs
        )
        self.epoch = max(self.epoch, epochs)

    def mold_inputs(self, images):
        """Takes a list of images and modifies them to the format expected
        as an input to the neural network.
        images: List of image matricies [height,width,depth,channels]. Images can have
            different sizes.

        Returns 3 Numpy matricies:
        molded_images: [N, h, w, d, 1]. Images resized and normalized.
        image_metas: [N, length of meta data]. Details about each image.
        windows: [N, (y1, x1, z1, y2, x2, z2)]. The portion of the image that has the
            original image (padding excluded).
        """
        molded_images = []
        for image in images:
            # Resize image to fit the model expected size
            # TODO: move resizing to mold_image()

            molded_image = mold_image(image, self.config)

            molded_images.append(molded_image)

        # Pack into arrays
        molded_images = np.stack(molded_images)
        return molded_images

    def unmold_detections(self, detections, image_shape, window, config):
        """Reformats the detections of one image from the format of the neural
        network output to a format suitable for use in the rest of the
        application.

        detections: [N, (y1, x1, z1, y2, x2, z2, class_id, score)]
        image_shape: [height, width, depth, channels] Original size of the image before resizing
        window: [y1, x1, z1, y2, x2, z2] Box in the image where the real image is
                excluding the padding.

        Returns:
        boxes: [N, (y1, x1, z1, y2, x2, z2)] Bounding boxes in pixels
        class_ids: [N] Integer class IDs for each bounding box
        scores: [N] Float probability scores of the class_id
        """
        # How many detections do we have?
        # Detections array is padded with zeros. Find the first class_id == 0.
        zero_ix = np.where(detections[:,6] == config.NUM_CLASSES-1)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]
        # Extract boxes, class_ids, scores, and class-specific masks
        boxes = detections[:N, :6]
        print  "boxes in unmold detections", boxes
        class_ids = detections[:N, 6].astype(np.int32)
        scores = detections[:N, 7]


        # Filter out detections with zero area. Often only happens in early
        # stages of training when the network weights are still a bit random.
        exclude_ix = np.where((boxes[:, 3] - boxes[:, 0]) * (boxes[:, 4] - boxes[:, 1]) * (boxes[:, 5] - boxes[:, 2]) <= 0)[0]
        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            class_ids = np.delete(class_ids, exclude_ix, axis=0)
            scores = np.delete(scores, exclude_ix, axis=0)
            N = class_ids.shape[0]


        # Compute scale and shift to translate coordinates to image domain.
        h_scale = image_shape[0] / (window[3] - window[0])
        w_scale = image_shape[1] / (window[4] - window[1])
        d_scale = image_shape[2] / (window[5] - window[2])
        scale = min(h_scale, w_scale, d_scale)
        shift = window[:3]  # y, x, z
        scales = np.array([scale, scale, scale, scale, scale, scale])
        shifts = np.array([shift[0], shift[1], shift[2], shift[0], shift[1], shift[2]])

        # Translate bounding boxes to image domain
        boxes = np.multiply(boxes - shifts, scales).astype(np.int32)



        return boxes, class_ids, scores

    def detect(self, images, verbose=0, config=None):
        """Runs the detection pipeline.

        images: List of images, potentially of different sizes.

        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        masks: [H, W, N] instance binary masks
        """
        assert self.mode == "inference", "Create model in inference mode."

        if verbose:
            log("Processing {} images".format(len(images)))
            for image in images:
                log("image", image)
        # Mold inputs to format expected by the neural network
        molded_images = self.mold_inputs(images)
        if verbose:
            log("molded_images", molded_images)
        # Run object detection
        reconstructed_images = self.keras_model.predict([molded_images], verbose=0)

        # Process detections
        results = []

        for i, reconstructed_image in enumerate(reconstructed_images):

            results.append({
                "image": reconstructed_image
            })
        return results

    def ancestor(self, tensor, name, checked=None):
        """Finds the ancestor of a TF tensor in the computation graph.
        tensor: TensorFlow symbolic tensor.
        name: Name of ancestor tensor to find
        checked: For internal use. A list of tensors that were already
                 searched to avoid loops in traversing the graph.
        """
        checked = checked if checked is not None else []
        # Put a limit on how deep we go to avoid very long loops
        if len(checked) > 500:
            return None
        # Convert name to a regex and allow matching a number prefix
        # because Keras adds them automatically
        if isinstance(name, str):
            name = re.compile(name.replace("/", r"(\_\d+)*/"))

        parents = tensor.op.inputs
        for p in parents:
            if p in checked:
                continue
            if bool(re.fullmatch(name, p.name)):
                return p
            checked.append(p)
            a = self.ancestor(p, name, checked)
            if a is not None:
                return a
        return None

    def find_trainable_layer(self, layer):
        """If a layer is encapsulated by another layer, this function
        digs through the encapsulation and returns the layer that holds
        the weights.
        """
        if layer.__class__.__name__ == 'TimeDistributed':
            return self.find_trainable_layer(layer.layer)
        return layer

    def get_trainable_layers(self):
        """Returns a list of layers that have weights."""
        layers = []
        # Loop through all layers
        for l in self.keras_model.layers:
            # If layer is a wrapper, find inner trainable layer
            l = self.find_trainable_layer(l)
            # Include layer if it has weights
            if l.get_weights():
                layers.append(l)
        return layers

    def run_graph(self, images, outputs):
        """Runs a sub-set of the computation graph that computes the given
        outputs.

        outputs: List of tuples (name, tensor) to compute. The tensors are
            symbolic TensorFlow tensors and the names are for easy tracking.

        Returns an ordered dict of results. Keys are the names received in the
        input and values are Numpy arrays.
        """
        model = self.keras_model

        # Organize desired outputs into an ordered dict
        outputs = OrderedDict(outputs)
        for o in outputs.values():
            assert o is not None

        # Build a Keras function to run parts of the computation graph
        inputs = model.inputs
        if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
            inputs += [K.learning_phase()]
        kf = K.function(model.inputs, list(outputs.values()))

        # Run inference
        molded_images, image_metas, windows = self.mold_inputs(images)
        # TODO: support training mode?
        # if TEST_MODE == "training":
        #     model_in = [molded_images, image_metas,
        #                 target_rpn_match, target_rpn_bbox,
        #                 gt_boxes, gt_masks]
        #     if not config.USE_RPN_ROIS:
        #         model_in.append(target_rois)
        #     if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
        #         model_in.append(1.)
        #     outputs_np = kf(model_in)
        # else:

        model_in = [molded_images, image_metas]
        if model.uses_learning_phase and not isinstance(K.learning_phase(), int):
            model_in.append(0.)
        outputs_np = kf(model_in)

        # Pack the generated Numpy arrays into a a dict and log the results.
        outputs_np = OrderedDict([(k, v) for k, v in zip(outputs.keys(), outputs_np)])
        for k, v in outputs_np.items():
            log(k, v)
        return outputs_np


############################################################
#  Data Formatting
############################################################

def compose_image_meta(image_id, image_shape, window, active_class_ids):
    """Takes attributes of an image and puts them in one 1D array. Use
    parse_image_meta() to parse the values back.

    image_id: An int ID of the image. Useful for debugging.
    image_shape: [height, width, channels]
    window: (y1, x1, y2, x2) in pixels. The area of the image where the real
            image is (excluding the padding)
    active_class_ids: List of class_ids available in the dataset from which
        the image came. Useful if training on images from multiple datasets
        where not all classes are present in all datasets.
    """
    meta = np.array(
        [image_id] +            # size=1
        list(image_shape) +     # size=3
        list(window) +          # size=4 (y1, x1, y2, x2) in image cooredinates
        list(active_class_ids)  # size=num_classes
    )
    return meta


# Two functions (for Numpy and TF) to parse image_meta tensors.
def parse_image_meta(meta):
    """Parses an image info Numpy array to its components.
    See compose_image_meta() for more details.
    """
    image_id = meta[:, 0]
    image_shape = meta[:, 1:5]
    window = meta[:, 5:11]   # (y1, x1, z1, y2, x2, z2) window of image in in pixels
    active_class_ids = meta[:, 11:]
    return image_id, image_shape, window, active_class_ids


def parse_image_meta_graph(meta):
    """Parses a tensor that contains image attributes to its components.
    See compose_image_meta() for more details.

    meta: [batch, meta length] where meta length depends on NUM_CLASSES
    """
    image_id = meta[:, 0]
    image_shape = meta[:, 1:4]
    window = meta[:, 4:8]
    active_class_ids = meta[:, 8:]
    return [image_id, image_shape, window, active_class_ids]


def mold_image(images, config):
    """Takes RGB images with 0-255 values and subtraces
    the mean pixel and converts it to float. Expects image
    colors in RGB order.
    """
    images = images.astype(np.float32)
    return images / np.max(images)
    #return images.astype(np.float32) - config.MEAN_PIXEL


def unmold_image(normalized_images, config):
    """Takes a image normalized with mold() and returns the original."""
    return (normalized_images * 255).astype(np.uint8)



