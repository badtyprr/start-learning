# Preprocessors that apply matrix shaping operations

# Python Packages
from typing import Union
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
        self.name = 'ImageToTensorPreprocessor'
        if dataFormat:
            self.name = self.name + '_' + str(dataFormat)
        # store the image data format
        self.dataFormat = dataFormat

    def preprocess_image(self, image: np.array) -> np.array:
        # Rearrange the dimensions of the image and flatten
        return img_to_array(image, data_format=self.dataFormat)

class CropPreprocessor(ImagePreprocessor):
    def __init__(
            self,
            width: Union[int, float],
            height: Union[int, float],
            split: bool=False):
        """
        Crops an image to width and height.
        If split is specified, splits the image into 5 images: 4 corners and center.
        :param width: Union[int, float] type representing the [horizontal resolution, horizontal % of the frame to include in crop]
        :param height: Union[int, float] type representing the [vertical resolution, vertical % of the frame to include in crop]
        :param split: bool type indicating whether to split the image into crops
        """
        # Absolute resolution or percentage-based dimensions
        if isinstance(width, int):
            self._absolute = True
        else:
            self._absolute = False
        self.width = width
        self.height = height
        # Crop 5 or 1
        self.split = split

        def crop(self, image: np.array) -> np.array:
            crops = []
            # Center crop
            (image_height, image_width) = image.shape[:2]
            # Absolute pixel dimensions
            if self._absolute:
                crop_width = self.width
                crop_height = self.height
            # Percentage-based pixel dimensions
            else:
                crop_width = self.width * image_width
                crop_height = self.height * image_height
            dW = int((image_width - crop_width) / 2.0)
            dH = int((image_height - crop_height) / 2.0)
            assert dW >= 0, 'crop width ({}) is greater than the original image width ({})'.format(crop_width, image_width)
            assert dH >= 0, 'crop height ({}) is greater than the original image height({})'.format(crop_height, image_height)
            crops.append(image[dH:image_height - dH, dW:image_width - dW])
            # Also split into 4 corners
            if self.split:
                # Upper left
                crops.append(image[0:crop_height, 0:crop_width])
                # Upper right
                crops.append(image[0:crop_height, image_width-crop_width:image_width])
                # Lower left
                crops.append(image[image_height-crop_height:image_height, 0:crop_width])
                # Lower right
                crops.append(image[image_height-crop_height:image_height, image_width-crop_width:image_width])
            # Return 1 or 5 crops
            return np.array(crops)

        def preprocess_image(self, image: np.array) -> np.array:
            return self.crop(image)

class ResizePreprocessor(ImagePreprocessor):
    def __init__(
            self,
            width: int,
            height: int,
            interpolation: int=cv2.INTER_AREA,
            aspect_preserving: bool=False):
        super().__init__()
        self.name = 'ResizePreprocessor_{}x{}'.format(width, height)
        self.width = width
        self.height = height
        self.interpolation = interpolation
        self.aspect_preserving = aspect_preserving

    def _resize(
            self, image: np.array,
            width: int=0, height: int=0,
            interpolation: int=cv2.INTER_AREA,
            keep_aspect: bool=False):
        """
        Resizes an image to widthxheight with interpolation and the
        option to keep aspect ratio
        :param image: The image to resize
        :param width: Width in pixels
        :param height: Height in pixels
        :param interpolation: OpenCV interpolation type, defaults to INTER_AREA
        :return: The resized image
        """
        # Without a width or height specification, return input
        if not (width or height):
            return image

        # Preprocess image to preserve aspect before resize
        if self.aspect_preserving:
            (h, w) = image.shape[:2]
            dW = 0
            dH = 0
            # Shortest dimension is width?
            if w < h:
                # Resize to the width dimension
                scale_factor = float(width) / w
                image = cv2.resize(
                    image,
                    dsize=None,
                    fx=scale_factor,
                    fy=scale_factor
                )
                dH = int((image.shape[0] - height) / 2.0)
            # Shortest dimension is height
            else:
                # Resize to the height dimension
                scale_factor = float(height) / h
                image = cv2.resize(
                    image,
                    dsize=None,
                    fx=scale_factor,
                    fy=scale_factor
                )
                dW = int((image.shape[1] - width) / 2.0)
            # Center crop
            image = image[dH:h - dH, dW:w - dW]
        # Resize the image according to the provided width and height
        return cv2.resize(
            image,
            (width, height),
            interpolation=interpolation
        )

    def preprocess(self, data: np.array):
        return self._resize(
            data,
            self.width, self.height,
            self.interpolation
        )


class ColorSpacePreprocessor(ImagePreprocessor):
    def __init__(self, conversion=cv2.COLOR_BGR2GRAY):
        super().__init__()
        self.name = 'ColorSpacePreprocessor_{}'.format(conversion)
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
