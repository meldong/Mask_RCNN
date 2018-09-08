# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 07:02:58 2018

@author: meldo
"""

import cv2
import datetime
import json
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import re
import shutil
import skimage.draw
import sys
import time
import urllib.request

# Root directory of the project
ROOT_DIR = os.path.abspath("./")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
import mrcnn.model as modellib
from mrcnn import utils, visualize

# Directory to store train and val data
DEFAULT_DATASETS_DIR = os.path.join(ROOT_DIR, "datasets")

# Local path to ResNet101 trained COCO weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "_init_/mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_WEIGHTS_PATH):
    """Download COCO trained weights from Releases."""
    COCO_WEIGHTS_URL = "https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5"
    print("Downloading pretrained model to " + COCO_WEIGHTS_PATH + " ...")
    with urllib.request.urlopen(COCO_WEIGHTS_URL) as resp, open(COCO_WEIGHTS_PATH, 'wb') as out:
        shutil.copyfileobj(resp, out)
    print("... done downloading pretrained model!")

# Local path to ResNet50 trained IMAGENET weights file
IMAGENET_WEIGHTS_PATH = os.path.join(ROOT_DIR, "_init_/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(IMAGENET_WEIGHTS_PATH):
    """Download IMAGENET trained weights from Releases."""
    IMAGENET_WEIGHTS_URL = "https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
    print("Downloading pretrained model to " + IMAGENET_WEIGHTS_PATH + " ...")
    with urllib.request.urlopen(IMAGENET_WEIGHTS_URL) as resp, open(IMAGENET_WEIGHTS_PATH, 'wb') as out:
        shutil.copyfileobj(resp, out)
    print("... done downloading pretrained model!")

# Local path to MobileNet trained IMAGENET weights file
MNV1_WEIGHTS_PATH = os.path.join(ROOT_DIR, "_init_/mobilenet_1_0_224_tf_no_top.h5")

# Directory to save logs and checkpoints, if not provided through flag --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


############################################################
#  Configurations
############################################################

class MyConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + Object

    # Input image resizing
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class MyDataset(utils.Dataset):

    def load_dataset(self, dataset_dir, subset):
        """Load a subset of the dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class(args.dataset, 1, "alpha")
        # self.add_class(args.dataset, 2, "beta")
        # self.add_class(args.dataset, 3, "gamma")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)
            polygons = [r['shape_attributes'] for r in a['regions'].values()]

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                args.dataset,
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != args.dataset:
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == args.dataset:
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


############################################################
#  Training
############################################################

def train(model):
    """Train the model."""
    # Dataset folder
    dataset_dir = os.path.join(DEFAULT_DATASETS_DIR, args.dataset)

    # Training dataset
    print(">> Prepare training dataset...")
    dataset_train = MyDataset()
    dataset_train.load_dataset(dataset_dir, "train")
    dataset_train.prepare()
    print("Image Count: {}".format(len(dataset_train.image_ids)))
    print("Class Count: {}".format(dataset_train.num_classes))
    for i, info in enumerate(dataset_train.class_info):
        print("{:3}. {:50}".format(i, info['name']))
    print("Random samples:")
    image_ids = np.random.choice(dataset_train.image_ids, 3)
    for image_id in image_ids:
        image = dataset_train.load_image(image_id)
        mask, class_ids = dataset_train.load_mask(image_id)
        bbox = utils.extract_bboxes(mask)
        print("image_id ", image_id, dataset_train.image_reference(image_id))
        modellib.log("image", image)
        modellib.log("mask", mask)
        modellib.log("class_ids", class_ids)
        modellib.log("bbox", bbox)
        visualize.display_top_masks(image, mask, class_ids,
                                    dataset_train.class_names)

    # Validation dataset
    print(">> Prepare validation dataset...")
    dataset_val = MyDataset()
    dataset_val.load_dataset(dataset_dir, "val")
    dataset_val.prepare()
    print("Image Count: {}".format(len(dataset_val.image_ids)))
    print("Class Count: {}".format(dataset_val.num_classes))
    for i, info in enumerate(dataset_val.class_info):
        print("{:3}. {:50}".format(i, info['name']))
    print("Random samples:")
    image_ids = np.random.choice(dataset_val.image_ids, 3)
    for image_id in image_ids:
        image = dataset_val.load_image(image_id)
        mask, class_ids = dataset_val.load_mask(image_id)
        bbox = utils.extract_bboxes(mask)
        print("image_id ", image_id, dataset_val.image_reference(image_id))
        modellib.log("image", image)
        modellib.log("mask", mask)
        modellib.log("class_ids", class_ids)
        modellib.log("bbox", bbox)
        visualize.display_top_masks(image, mask, class_ids,
                                    dataset_val.class_names)

	# Image Augmentation (refer to coco.py for how to use)
    # Right/Left flip 50% of the time
    # augmentation = imgaug.augmenters.Fliplr(0.5)

    # *** This training schedule is an example. Update to your needs ***

    # Training network heads
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')

    # Save weights
    print("Save weights")
    model_path = os.path.join(DEFAULT_LOGS_DIR, "mask_rcnn_heads.h5")
    model.keras_model.save_weights(model_path)

    # Fine tune all layers
    print("Fine tune all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=30,
                layers='all')

    # Save weights
    print("Save weights")
    model_path = os.path.join(DEFAULT_LOGS_DIR, "mask_rcnn_all.h5")
    model.keras_model.save_weights(model_path)


############################################################
#  Evaluation
############################################################

def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash

def evaluate(model, image_path=None, video_path=None):
    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print(">> Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
        print(">> Saved to ", file_name)
    elif video_path:
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)
        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))
        # Loop through frames
        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
        print(">> Saved to ", file_name)
    else:
        # Test on a random image
#        image_id = random.choice(dataset_val.image_ids)
#        original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
#            modellib.load_image_gt(dataset_val, inference_config,image_id,
#                                   use_mini_mask=False)
#        modellib.log("original_image", original_image)
#        modellib.log("image_meta", image_meta)
#        modellib.log("gt_class_id", gt_class_id)
#        modellib.log("gt_bbox", gt_bbox)
#        modellib.log("gt_mask", gt_mask)
#
#        visualize.display_instances(original_image, gt_bbox, gt_mask,
#                                    gt_class_id, dataset_train.class_names,
#                                    figsize=(8, 8))
#
#        r = model.detect([original_image], verbose=1)[0]
#        visualize.display_instances(original_image, r['rois'], r['masks'],
#                                    r['class_ids'], dataset_val.class_names,
#                                    r['scores'], ax=get_ax())
        print("Done!")


############################################################
#  Main
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect objects.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'eval'")
    parser.add_argument('--dataset', required=True,
                        metavar="name of dataset",
                        help='Name of the dataset')
    parser.add_argument('--backbone', required=False,
                        default="resnet101",
                        metavar="<backbone>",
                        help='Feature Pyramid Network backbone type')
    parser.add_argument('--weights', required=True,
                        metavar="path to weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="path to logs",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or url to image",
                        help='Image to detect objects on')
    parser.add_argument('--video', required=False,
                        metavar="path or url to video",
                        help='Video to detect objects on')
    args = parser.parse_args()

    # Validate arguments
    assert args.command, "Argument command is required for train or eval"
    assert args.dataset, "Argument --dataset is required for train or eval"
    assert args.weights, "Argument --weights is required for train or eval"
    # if args.command == "eval":
        # assert args.image or args.video,\
               # "Argument --image or --video is required for evaluation"

    print("Command: ", args.command)
    print("Dataset: ", args.dataset)
    print("Backbone: ", args.backbone)
    print("Weights: ", args.weights)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = MyConfig()
    else:
        class InferenceConfig(MyConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()

    # Configure dataset name
    config.NAME = args.dataset

    # Configure backbone architecture
    if args.backbone.lower() == "resnet50":
        config.BACKBONE = "resnet50"
    elif args.backbone.lower() == "mobilenet224v1":
        config.BACKBONE = "mobilenet224v1"
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
    elif args.weights.lower() == "imagenet":
        weights_path = IMAGENET_WEIGHTS_PATH
    elif args.weights.lower() == "mnv1":
        weights_path = MNV1_WEIGHTS_PATH
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    else:
        weights_path = args.weights

    # Load weights
    print(">> Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "eval":
        evaluate(model, image_path=args.image, video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'eval'".format(args.command))
