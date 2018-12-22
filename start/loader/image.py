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
    """
    ImageCachedDataset reads images from a directory or CSV
    :param dataset_path: Path type representing the path to a directory or CSV
    :param subdirectory: Path type used with CSVs if the images are in a directory different from the CSV file
    :param preprocessors: list type representing Preprocessor types to operate on the images in sequential order
    :param cache_path: Path type representing the path to a caching directory
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _csv_handler(self, properties: dict=None) -> (np.array, np.array):
        super()._csv_handler(properties)
        print('[INFO] Retrieving dataset from dataframe:\n{}'.format(self.dataframe.describe()))
        # Implements data loading
        return self._cached_csv_handler(properties)

    def _cached_csv_handler(self, properties: dict=None) -> (np.array, np.array):
        i = 0
        data = []
        labels = []
        path_to, filename = os.path.split(self.dataset_path)
        for index, row in self._dataframe.iterrows():
            if row['Id'] in self._ignored_labels:
                continue
            image_path = Path(os.path.join(path_to, self.subdirectory, row['Image']))
            image = self._cached_retrieve(image_path)
            if image is None:
                print('[WARNING] {} could not be read, skipping...'.format(image_path))
                continue
            # Add to data and labels
            data.append(image)
            labels.append(row['Id'])
            i += 1

            # Update every 'verbosity' images
            verbosity = properties.get('verbosity', -1)
            if verbosity > 0 and i % verbosity == 0:
                print("[INFO] processed {}/{} images".format(i, self._dataframe.shape[0]))

        return np.array(data), np.array(labels)

    def _cached_directory_handler(self, verbosity: int) -> (np.array, np.array):
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
            if label in self._ignored_labels:
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
                        if image is None:
                            continue
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

    def _cached_retrieve(self, key: Path) -> np.array:
        """
        Retrieves a data sample from the datastore using key, preprocesses the data and caches the result
        :param key: str type representing the unique identifier that can store the data sample
        :return: numpy array representing an image
        """
        key_path, filename = os.path.split(key)
        label_directories = os.path.relpath(key_path, os.path.commonprefix([self.dataset_path, key_path]))
        label_path = os.path.join(self.cache_path, self.subdirectory, label_directories)
        preprocessor_path = label_path
        for p in self.preprocessors:
            preprocessor_path = os.path.join(preprocessor_path, str(p))
        image = cv2.imread(os.path.join(preprocessor_path, filename), cv2.IMREAD_UNCHANGED)
        # If the image doesn't exist, then preprocess the original image and cache results along the way
        if image is None:
            try:
                os.makedirs(preprocessor_path)
            except FileExistsError:
                pass
            image = cv2.imread(str(key), cv2.IMREAD_UNCHANGED)
            if image is None:
                return image
            preprocessor_path = label_path
            for p in self.preprocessors:
                preprocessor_path = os.path.join(preprocessor_path, str(p))
                image = p.preprocess(image)
                # Cache result
                self._cached_store(
                    key=Path(os.path.join(preprocessor_path, filename)),
                    data=image
                )
        # Check that the shape is at least rank 3
        if len(image.shape) < 3:
            image = image[:,:,np.newaxis]
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
            for root, dirs, files in os.walk(self.dataset_path):
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
                        # TODO: Remove cached images too, maybe a _cached_remove and _remove methods?
                        # TODO: For that matter, an _add and _cached_add ?
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
                if not properties.get('dry-run', False):
                    os.remove(item_path)
            except FileNotFoundError:
                print('[WARNING] Could not find file: {}'.format(item_path))

    def _cached_load(self, properties: dict=None) -> (np.array, np.array):
        """
        Loads a dataset, caching data as they are preprocessed.
        If cached preprocessed data exists, load it instead of preprocessing the original data.
        :param properties: dict type containing all relevant flags and variables for the handler
        :return: numpy arrays of the training and test sets
        """
        if not properties:
            properties = {}
        if type(self.dataset_path) in [str, Path]:
            if os.path.isdir(self.dataset_path):
                return self.handlers[type(self.dataset_path)](properties)
            elif os.path.isfile(self.dataset_path):
                path_to, filename = os.path.split(self.dataset_path)
                basefilename, ext = os.path.splitext(filename)
                return self.handlers[ext[1:]](properties)

