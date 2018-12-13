# Writers that write to a file

# Python packages
import os
from pathlib import Path
from abc import ABC, abstractmethod
# User Packages
from .base import Writer

class FileWriter(ABC, Writer):
    def __init__(self, filepath: Path):
        # NOTE: I don't think ABC requires any arguments to be passed to it?
        ABC.__init__(self)
        Writer.__init__(self)
        # Check for path existence
        if os.path.exists(filepath):
            raise ValueError('The filepath already exists: {}'.format(filepath))
        self.filepath = filepath

    @abstractmethod
    def flush(self):
        pass


class DatasetFileWriter(FileWriter):
    def __init__(self, dimensions: tuple):
        super().__init__()
        # NOTE: Every dataset will have some data of a particular dimension,
        # the methods of which to write such data will be up to the inheriting
        # class.
        if not dimensions:
            raise ValueError('dimensions must not be empty')
        self.dimensions = dimensions

    @abstractmethod
    def flush(self):
        pass

    @abstractmethod
    def close(self):
        pass