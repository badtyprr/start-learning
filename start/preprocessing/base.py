# Base Preprocessor class

# Python Packages
from abc import ABC, abstractmethod
# 3rd Party Packages
import numpy as np

# Thanks: http://masnun.rocks/2017/04/15/interfaces-in-python-protocols-and-abcs/
class Preprocessor(ABC):
    def __init__(self):
        super().__init__()
        self.name = 'Preprocessor'

    @abstractmethod
    def preprocess(self, data):
        return data

    def __str__(self):
        return self.name


class ImagePreprocessor(Preprocessor):
    def __init__(self):
        super().__init__()
        self.name = 'ImagePreprocessor'

    def preprocess(self, data: np.array) -> np.array:
        # Do nothing by default
        return data

