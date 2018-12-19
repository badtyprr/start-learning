# Loaders for datasets

# Python Packages
import os
from pathlib import Path
from shutil import rmtree
# 3rd Party Packages
import cv2
import numpy as np
import pandas as pd
# User Packages
from .base import CachedDataset
from .pandas import CSVDatasetMixin
from ..utils import quantized_histogram


class ImageCachedDataset(CachedDataset, CSVDatasetMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _cached_directory_handler(self, verbosity: int):
        """
        When dataset path is a directory, treat it as a folder structured labeling system
        :param directory: Path type representing a directory of data samples
        :return: (training data, test data)
        """
        # Data holds image data
        data = []
        # Label holds text label
        labels = []

        # Read data from image paths
        for label in os.listdir(self.dataset_path):
            if label in self._cache_folder_name:
                continue
            i = 0
            # Read image and get path
            print('[INFO] Processing label: {}'.format(label))
            try:
                label_path = os.path.join(self.dataset_path, label)
                for root, dirs, files in os.walk(label_path):
                    for file in files:
                        image = self._cached_retrieve(
                            Path(os.path.join(root, file))
                        )

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

    def _cached_retrieve(self, key: Path):
        """
        Retrieves a data sample from the datastore using key, preprocesses the data and caches the result
        :param key: str type representing the unique identifier that can store the data sample
        :return: numpy array representing an image
        """
        key_path, filename = os.path.split(key)
        label_directories = os.path.relpath(key_path, os.path.commonprefix([self.dataset_path, key_path]))
        label_path = os.path.join(self.cache_path, label_directories)
        preprocessor_path = label_path
        for p in self.preprocessors:
            preprocessor_path = os.path.join(preprocessor_path, str(p))
        directory, filename = os.path.split(key)
        image = cv2.imread(os.path.join(preprocessor_path, filename))
        # If the image doesn't exist, then preprocess the original image and cache results along the way
        if image is None:
            try:
                os.makedirs(preprocessor_path)
            except FileExistsError:
                pass
            image = cv2.imread(str(key))
            preprocessor_path = label_path
            for p in self.preprocessors:
                preprocessor_path = os.path.join(preprocessor_path, str(p))
                image = p.preprocess(image)
                # Cache result
                self._cached_store(
                    key=Path(os.path.join(preprocessor_path, filename)),
                    data=image
                )
        # Return the final preprocessed image
        return image

    def _cached_store(self, key: Path, data):
        """
        Stores an image in the datastore.
        :param key: Path type representing the filepath
        :param data: np.array representing an image
        """
        cv2.imwrite(str(key), data)

    def _cached_clean(self, properties: dict):
        """
        Cleans an image dataset with respect to properties' modes
        :param properties: Filter images by one or more filtering methods
            properties['input']: dataset directory
            properties['duplicate']: Set to True to enable removing duplicates
            properties['blurry'] = Set to True to enable removing blurry images (based on the variance of the laplacian, NOT implemented yet)
            properties['detector'] = Set to True to enable removing images that do not have detections of a particular class (uses pretrained detector, NOT implemented yet)
            properties['cache'] = Set to True to delete the cache folder
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
                        continue
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

        if properties.get('cache', False):
            rmtree(self.cache_path)

        # Combined removal
        for item_path in setRemove:
            try:
                os.remove(item_path)
            except FileNotFoundError:
                print('[WARNING] Could not find file: {}'.format(item_path))

    def _cached_load(self, verbosity: int = -1) -> (np.array, np.array):
        """
        Loads a dataset, caching data as they are preprocessed.
        If cached preprocessed data exists, load it instead of preprocessing the original data.
        :param dataset_path: Path type representing the path to the dataset's base directory
        :param verbosity: Prints every [verbosity] data loads
        :return: numpy arrays of the training and test sets
        """

        return self.handlers[type(self.dataset_path)](verbosity)




