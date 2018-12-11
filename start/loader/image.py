# Loaders for datasets

# Python Packages
import os
from pathlib import Path
from typing import Union
# 3rd Party Packages
import cv2
import numpy as np
import pandas as pd
# User Packages
from .base import Dataset
from .pandas import PandasDatasetMixin
from ..utils import quantized_histogram

# NOTE: For a Titan X with 12GB of memory,
# a 224x224x3 image size will only allow ~79K images per batch
# For 16 classes, that's 4,966 images per class

class ImageDataset(Dataset, PandasDatasetMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load(self, dataset_path: Union[str, Path], verbosity=-1):
        data = []
        labels = []

        if isinstance(dataset_path, Path) or isinstance(dataset_path, str):
            # If an image directory is given, load images by walking through
            if os.path.isdir(dataset_path):
                # Read data from image paths
                for label in os.listdir(dataset_path):
                    i = 0
                    # Read image and get path
                    print('[INFO] Processing label: {}'.format(label))
                    try:
                        label_path = os.path.join(dataset_path, label)
                        for image_path in os.listdir(label_path):
                            image = cv2.imread(os.path.join(label_path, image_path))
                            # Only process images that can be read
                            if image is not None:
                                # Pre-process image
                                for p in self.preprocessors:
                                    image = p.preprocess(image)

                                # Add to data and labels
                                data.append(image)
                                labels.append(label)
                                i += 1

                            # Update every 'verbosity' images
                            if verbosity > 0 and i % verbosity == 0:
                                print("[INFO] processed {} {} images".format(i, label))
                    # Not a directory, and therefore, not a label
                    except (FileNotFoundError, NotADirectoryError):
                        pass

                return np.array(data), np.array(labels)
            # If a pandas dataframe is loaded, load images from paths
            elif os.path.isfile(dataset_path):
                self.load_dataframe(dataset_path)
                # DataFrame is in self.dataframe
                # NOTE: Kaggle Whales will take 4,529,677,488 bytes of RAM if downsized to 244x244x3


            else:
                raise ValueError('dataset path does not exist')

        else:
            raise ValueError('dataset must point to an image directory or pandas dataframe file')

    def clean(self, properties: dict):
        """

        :param properties: Filter images by one or more filtering methods
            properties['input']: dataset directory
            properties['duplicate']: Set to True to enable removing duplicates
            properties['blurry'] = Set to True to enable removing blurry images (based on the variance of the laplacian, NOT implemented yet)
            properties['detector'] = Set to True to enable removing images that do not have detections of a particular class (uses pretrained detector, NOT implemented yet)
        """

        # Data structure => dict[(filesize, histogram[0...63])],
        #   where histogram is a tuple of a quantized 6-bit RGB space (RRGGBB)
        dDuplicate = {}
        setRemove = set()

        if properties.get('duplicate', False):
            for root, dirs, files in os.walk(properties['input']):
                for file in files:
                    filepath = os.path.join(root, file)
                    # Read file stats (size)
                    stats = os.stat(filepath)
                    # Read image file
                    image = cv2.imread(filepath, cv2.IMREAD_REDUCED_COLOR_8)
                    # If the image cannot be opened, remove it
                    if image is None:
                        # Mark for removal
                        setRemove.add(filepath)
                    hist = quantized_histogram(image)
                    hashable_key = (stats.st_size, tuple(hist))
                    # Try to add to existing key
                    try:
                        dDuplicate[hashable_key].append(filepath)
                    # Initialize new key
                    except KeyError:
                        dDuplicate[hashable_key] = [filepath]

            # Delete duplicates (but keep the first one)
            for key in [k for k in dDuplicate.keys() if len(dDuplicate[k]) > 1]:
                for i in range(1, len(dDuplicate[key])):
                    print('[INFO] Marking duplicate for removal: {}'.format(dDuplicate[key][i]))
                    setRemove.add(dDuplicate[key][i])

        if properties.get('blurry', False):
            raise NotImplementedError('Filtering by blurriness is not yet implemented')

        if properties.get('detector', False):
            raise NotImplementedError('Filtering by detector is not yet implemented')

        # Combined removal
        for item_path in setRemove:
            try:
                os.remove(item_path)
            except FileNotFoundError:
                print('[WARNING] Could not find file: {}'.format(item_path))

