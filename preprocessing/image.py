# Preprocessors that apply matrix shaping operations

# 3rd Party Packages
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2
# User Packages
from .base import ImagePreprocessor


class ImageToTensorPreprocessor(ImagePreprocessor):
    def __init__(self, dataFormat=None):
        # Thanks: https://www.journaldev.com/15911/python-super
        super().__init__()
        # store the image data format
        self.dataFormat = dataFormat

    def preprocess(self, data: np.array):
        # Rearrange the dimensions of the image and flatten
        return img_to_array(data, data_format=self.dataFormat)


class ResizePreprocessor(ImagePreprocessor):
    def __init__(self, width, height, interpolation=cv2.INTER_AREA):
        super().__init__()
        self.width = width
        self.height = height

    def preprocess(self, data):
        # Resize the image
        return cv2.resize(
            data,
            (self.width, self.height),
            interpolation=self.interpolation
        )


class ColorSpacePreprocessor(ImagePreprocessor):
    def __init__(self, conversion=cv2.COLOR_BGR2GRAY):
        super().__init__()
        self.conversion = conversion

    def preprocess(self, data):
        if self.conversion is None:
            return data
        else:
            # Convert color space
            return cv2.cvtColor(
                data,
                self.conversion
            )

