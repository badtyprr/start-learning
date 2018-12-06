# Base class

from abc import ABC, abstractmethod


class Dataset(ABC):
    def __init__(self, preprocessors=None):
        if not preprocessors:
            self.preprocessors = []
        else:
            self.preprocessors = preprocessors

    @abstractmethod
    def load(self, datasetPaths, verbosity=-1):
        pass

