# Base Preprocessor class

# Python Packages
from abc import ABC, abstractmethod
# 3rd Party Packages
import numpy as np

# Thanks: http://masnun.rocks/2017/04/15/interfaces-in-python-protocols-and-abcs/
class Preprocessor(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def preprocess(self, data):
        return data


class ImagePreprocessor(Preprocessor):
    def __init__(self):
        super().__init__()

    def preprocess(self, data: np.array):
        # Do nothing by default
        return data

