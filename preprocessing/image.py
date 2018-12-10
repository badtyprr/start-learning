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
    def __init__(
            self,
            width: int,
            height: int,
            interpolation: int=cv2.INTER_AREA,
            aspect_preserving: bool=False):
        super().__init__()
        self.width = width
        self.height = height
        self.interpolation = interpolation
        self.aspect_preserving = aspect_preserving

    @classmethod
    def resize(
            self, image: np.array,
            width: int=0, height: int=0,
            interpolation: int=cv2.INTER_AREA,
            keep_aspect=False):
        """
        Resizes an image to widthxheight with interpolation and the
        option to keep aspect ratio
        :param image: The image to resize
        :param width: Width in pixels
        :param height: Height in pixels
        :param interpolation: OpenCV interpolation type, defaults to INTER_AREA
        :param keep_aspect: Keep aspect ratio or not
        :return: The resized image
        """
        # Without a width or height specification, return input
        if not (width or height):
            return image
        # Preprocess image to preserve aspect before resize
        elif keep_aspect:
            (h, w) = image.shape[:2]
            dW = 0
            dH = 0
            # Shortest dimension is width?
            if w < h:
                # Resize to the width dimension
                scaleFactor = float(width) / w
                image = cv2.resize(
                    image,
                    fx=scaleFactor,
                    fy=scaleFactor
                )
                dH = int((image.shape[0] - height) / 2.0)
            # Shortest dimension is height
            else:
                # Resize to the height dimension
                scaleFactor = float(height) / h
                image = cv2.resize(
                    image,
                    fx=scaleFactor,
                    fy=scaleFactor
                )
                dW = int((image.shape[1] - width) / 2.0)
            # Center crop
            (h, w) = image.shape[:2]
            image = image[dH:h - dH, dW:w - dW]
        # Resize the image according to the provided width and height
        return cv2.resize(
            image,
            (width, height),
            interpolation=interpolation
        )


    def preprocess(self, data: np.array):
        if self.aspect_preserving:
            return self.resize(data,
                               self.width, self.height,
                               self.interpolation,
                               keep_aspect=True)
        else:
            return self.resize(data,
                               self.width, self.height,
                               self.interpolation,
                               keep_aspect=False)


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
