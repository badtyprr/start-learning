# Base class for feature extraction

from abc import ABC, abstractmethod


class Feature(object):
    pass


class DescriptorFeature(Feature, ABC):
    def __init__(self):
        super().__init__()

    @staticmethod
    @abstractmethod
    def summarize(data):
        pass


class DimensionalReductionFeature(Feature, ABC):
    def __init__(self):
        super().__init__()

    @staticmethod
    @abstractmethod
    def reduce(data):
        pass

