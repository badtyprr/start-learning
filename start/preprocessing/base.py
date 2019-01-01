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

    @abstractmethod
    def preprocess_image(self, image: np.array) -> np.array:
        pass

    def preprocess(self, data: np.array) -> np.array:
        # Single image
        if len(data.shape) == 3:
            return self.preprocess_image(data)
        # Batch
        if len(data.shape) == 4:
            images = []
            for image in data:
                image = self.preprocess_image(image)
                images.append(image)
            return np.vstack(images)

