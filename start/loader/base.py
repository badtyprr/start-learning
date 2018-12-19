# Base class

# Python Packages
from abc import ABC, abstractmethod
from pathlib import Path
import os
# 3rd Party Packages
import numpy as np

class Dataset(ABC):
    def __init__(self, dataset_path: Path, preprocessors: list=None):
        if not preprocessors:
            self.preprocessors = []
        else:
            self.preprocessors = preprocessors

        self.handlers = {
            Path:   self._directory_handler,
            str:    self._directory_handler
        }

        self.dataset_path = dataset_path

    @abstractmethod
    def _directory_handler(self, directory: Path):
        pass

    @abstractmethod
    def retrieve(self, key):
        """
        Retrieves a data sample from the datastore using key and preprocesses the data
        :param key: str type representing the unique identifier that can retrieve the data sample
        :return: variable type data
        """
        pass

    @abstractmethod
    def store(self, key, data):
        """
        Stores a data sample on the datastore
        :param key: str type representing the unique identifier that can store the data sample
        :param data: variable data type to store
        """

    @abstractmethod
    def load(self, verbosity: int=-1) -> (np.array, np.array):
        """
        Loads a dataset and returns as numpy arrays
        :param verbosity: Prints status every [verbosity] data loads
        :return: numpy arrays of the training and test sets
        """
        # Check for path existence
        if not any(isinstance(self.dataset_path, t) for t in self.handlers.keys()):
            raise ValueError('No handler for dataset path')

    @abstractmethod
    def clean(self, properties: dict):
        """
        Cleans the dataset according to the properties given.
        :param properties: dict type representing the parameters that define how a clean is implemented
        """
        pass

class CachedDataset(Dataset):
    def __init__(self, cache_path: Path=None, *args, **kwargs):
        """
        A CachedDataset is assumed to be cached on disk at a cache path.
        A folder structure is made for each preprocessor and its preprocessor output cached.
        :param cache_path: Path type representing a directory to cache preprocessor outputs to
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)

        # Setup the cached directory path

        if cache_path:
            path_to, self._cache_folder_name = os.path.split(cache_path)
            self.cache_path = cache_path
        else:
            # Ignore when reading labels
            self._cache_folder_name = 'cache'
            # Defaults to dataset_path/cache
            self.cache_path = os.path.join(self.dataset_path, self._cache_folder_name)

        # Check for a handler
        if any(isinstance(self.cache_path, t) for t in self.handlers.keys()):
            if os.path.exists(self.cache_path):
                if not os.path.isdir(self.cache_path):
                    ValueError('cache_path must be a directory path')
            else:
                # Create directory path
                print('[WARNING] cache_path does not exist and will be created')
                os.makedirs(self.cache_path)
        else:
            raise ValueError('cache_path must be a directory path')

    @abstractmethod
    def _cached_directory_handler(self, directory: Path):
        """
        When dataset path is a directory, treat it as a folder structured labeling system
        :param directory: Path type representing a directory of data samples
        :return:
        """
        pass

    @abstractmethod
    def _cached_retrieve(self, key: Path):
        """
        Retrieves a data sample from the datastore using key, preprocesses the data and caches the result
        :param key: str type representing the unique identifier that can store the data sample
        :return: variable type data
        """
        pass

    @abstractmethod
    def _cached_store(self, key: Path, data):
        """
        Stores a data sample on the datastore. Same functionality as Dataset.store().
        :param key: str type representing the unique identifier that can store the data sample
        :param data: variable data type to store
        """
        pass

    @abstractmethod
    def _cached_load(self, verbosity: int=-1) -> (np.array, np.array):
        """
        Loads a dataset, caching data as they are preprocessed.
        If cached preprocessed data exists, load it instead of preprocessing the original data.
        :param dataset_path: Path type representing the path to the dataset's base directory
        :param verbosity: Prints every [verbosity] data loads
        :return: numpy arrays of the training and test sets
        """
        pass

    @abstractmethod
    def _cached_clean(self, properties: dict):
        """
        Cleans the dataset according to the properties given. Also cleans the cached images, if they exist.
        """
        pass

    def _directory_handler(self, directory: Path):
        self._cached_directory_handler(directory)

    def retrieve(self, key):
        self._cached_retrieve(key)

    def store(self, key, data):
        self._cached_store(key, data)

    def load(self, verbosity: int=-1):
        self._cached_load(verbosity)

    def clean(self, properties: dict):
        self._cached_clean(properties)