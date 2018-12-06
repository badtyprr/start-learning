# Loaders for datasets

# Python Packages
import os
# 3rd Party Packages
import cv2
import numpy as np
# User Packages
from .base import Dataset

class ImageDataset(Dataset):
    def __init__(self, preprocessors=None):
        super().__init__(preprocessors)

    def load(self, datasetPaths, verbosity=-1):
        data = []
        labels = []

        # Read data from image paths
        for (i, imagePath) in enumerate(datasetPaths):
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
                print("[INFO] processed {}/{}".format(i + 1, len(datasetPaths)))

        return np.array(data), np.array(labels)

