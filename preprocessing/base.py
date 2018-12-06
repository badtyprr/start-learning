# Base Preprocessor class

from abc import ABC, abstractmethod


# Thanks: http://masnun.rocks/2017/04/15/interfaces-in-python-protocols-and-abcs/
class Preprocessor(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def preprocess(self, data):
        return data


class ImagePreprocessor(Preprocessor):
    def __init__(self):
        super().__init__()

    def preprocess(self, data):
        # Do nothing by default
        return data

