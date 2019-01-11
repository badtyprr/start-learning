# Dataset (holds a catalog and responsible for control logic between classes)
#
# Catalog (handles data location, label, bounding box information, etc.)
# |- DirectoryCatalog (handles directory structured data)
# |- CSVCatalog (handles CSV structured data)
#
# ByteStream
# |- InputStream (handles the byte retrieval from the data source)
#   |- FileInputStream (handles byte streams from a file)
#   |- MemoryInputStream (handles byte streams from a bytearray)
#   ... there could be lots of others
# |- OutputStream (handles the byte storage to a data source)
#   |- FileOutputStream (handles byte streams to a file)
#     |- HDF5OutputStream (handles byte streams to an HDF5 file)
#   |- MemoryOutputStream (handles byte streams to memory)
#
# Decoder (handles decoding of a ByteStream into a format processable by TensorFlow)
# |- ImageDecoder (handles decoding into an image array, but how to handle different file formats when decoupled from the ByteStream?)
# ... there could be lots of others
#
# Preprocessor (handles preprocessing as a chain of preprocessors, taking in one or more samples per iteration)
# |- ResizePreprocessor
# |- ImageToTensorPreprocessor
# |- ColorSpacePreprocessor
# |- CropPreprocessor
# ... there could be lots of others
#
# Writer (handles the logistics of storing samples to a storage medium)
# |- HDF5Writer (handles logistics of writing data to an HDF5 file for larger datasets)
# |- DirectoryWriter (handles logistics of writing data in a directory hierarchy, usually coupled with a FileOutputStream or subclass of such)
# ... there could be lots of others
#
# Filter (filters what types of data a class can handle)

# Python Libraries
from abc import ABC, abstractmethod
from typing import List, Tuple, Callable, Union
from pathlib import Path
# 3rd Party Libraries
import numpy as np
# User Libraries
from ..preprocessing import Preprocessor

# Types
input_handler_t = Callable[..., Tuple[...]]
output_handler_t = Callable[..., int]
preprocessors_t = List[Preprocessor]
catalog_t = Union[str, Path]
content_t = Tuple[...]

class Dataset(ABC):
    def __init__(self, input_handler: input_handler_t, catalog: catalog_t, output_handler: output_handler_t=None, preprocessors: preprocessors_t=None):
        self._input_handler = input_handler
        self._catalog = catalog
        self._output_handler = output_handler
        self._preprocessors = preprocessors

    @property
    def input_handler(self) -> input_handler_t:
        return self._input_handler

    @input_handler.setter
    def input_handler(self, ih: input_handler_t):
        if not isinstance(ih, input_handler_t):
            raise ValueError('input_handler must be of type {}'.format(input_handler_t))
        self._input_handler = ih

    @property
    def output_handler(self) -> output_handler_t:
        return self._output_handler

    @output_handler.setter
    def output_handler(self, oh: output_handler_t):
        if not isinstance(oh, output_handler_t):
            raise ValueError('output_handler must be of type {}'.format(output_handler_t))
        self._output_handler = oh

    @property
    def preprocessors(self) -> preprocessors_t:
        return self._preprocessors

    @preprocessors.setter
    def preprocessors(self, p):
        if not isinstance(p, preprocessors_t):
            raise ValueError('preprocessors must be of type {}'.format(preprocessors_t))
        self._preprocessors = p

    @abstractmethod
    def load(self, opts: dict) -> object:
        """
        Generator that loads all or part of a dataset
        :return: subclass defined data
        """
        yield

    def store(self, opts: dict):
        """
        Stores all or part of a dataset (optional)
        """
        raise NotImplementedError('store() is not implemented yet!')

    def clean(self):
        """
        Frees up memory from loaded data (optional)
        """
        raise NotImplementedError('clean() is not implemented yet!')

    def reset(self):
        """
        Resets the load generator to the first dataset entry (optional)
        """
        raise NotImplementedError('reset() is not implemented yet!')

