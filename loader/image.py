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

class ImageDataset(Dataset, PandasDatasetMixin):
    def __init__(self, preprocessors=None):
        super().__init__(preprocessors)

    def load(self, dataset: Union[str, Path], verbosity=-1):
        data = []
        labels = []

        if isinstance(dataset, Path) or isinstance(dataset, str):
            # If an image directory is given, load images by walking through
            if os.path.isdir(dataset):
                # Read data from image paths
                for (i, imagePath) in enumerate(dataset):
                    # Read image and get path
                    image = cv2.imread(imagePath)
                    label = imagePath.split(os.path.sep)[-2]

                    # Only process images that can be read
                    if image is not None:
                        # Pre-process image
                        for p in self.preprocessors:
                            image = p.preprocess(image)

                        # Add to data and labels
                        data.append(image)
                        labels.append(label)

                    # Update every 'verbosity' images
                    if verbosity > 0 and i > 0 and (i + 1) % verbosity == 0:
                        print("[INFO] processed {}/{}".format(i + 1, len(dataset)))

                return np.array(data), np.array(labels)
            # If a pandas dataframe is loaded, load images from paths
            elif os.path.isfile(dataset):
                self.load_dataframe(dataset)
                # DataFrame is in self.dataframe
                # NOTE: Kaggle Whales will take 4,529,677,488 bytes of RAM if downsized to 244x244x3

            else:
                raise ValueError('dataset path does not exist')

        else:
            raise ValueError('dataset must point to an image directory or pandas dataframe file')


    def clean(self, properties):
        # Duplicates: properties['duplicates']
        # Blurry images from variance of laplacian: properties['blurry']
        # class using an existing detector: properties['detector']
        pass

