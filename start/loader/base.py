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
from ..utils import _validate_types, deferrableabstractmethod

# Types
content_t = Tuple[...]
input_handler_t = Callable[..., content_t]
output_handler_t = Callable[..., int]
preprocessors_t = List[Preprocessor]
catalog_t = Union[str, Path]

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
        _validate_types(ih=input_handler_t)
        self._input_handler = ih

    @property
    def output_handler(self) -> output_handler_t:
        return self._output_handler

    @output_handler.setter
    def output_handler(self, oh: output_handler_t):
        _validate_types(oh=output_handler_t)
        self._output_handler = oh

    @property
    def preprocessors(self) -> preprocessors_t:
        return self._preprocessors

    @preprocessors.setter
    def preprocessors(self, p):
        _validate_types(p=preprocessors_t)
        self._preprocessors = p

    @abstractmethod
    def load(self, opts: dict) -> content_t:
        """
        Generator that loads all or part of a dataset
        :return: subclass defined data
        """
        yield

    @deferrableabstractmethod
    def store(self, opts: dict=None):
        """
        Stores all or part of a dataset (optional)
        :param opts: dict type representing the parameters for the specific storage method
        :return:
        """
        pass

    @deferrableabstractmethod
    def free(self):
        """
        Frees up memory from loaded data (optional)
        """
        pass

    @deferrableabstractmethod
    def reset(self):
        """
        Resets the load generator to the first dataset entry (optional)
        """
        pass

    @deferrableabstractmethod
    def clean(self, opts: dict=None) -> int:
        """
        Cleans the dataset (e.g. removing duplicates, etc.)
        :param opts: dict type representing parameters for the type of clean required
        :return: int type representing the number of entries cleaned
        """
        pass

