# Base class

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union
import pandas as pd

class Dataset(ABC):
    def __init__(self, preprocessors=None):
        if not preprocessors:
            self.preprocessors = []
        else:
            self.preprocessors = preprocessors

    @abstractmethod
    def load(self, dataset_path: Union[str, Path], verbosity=-1):
        pass

    @abstractmethod
    @staticmethod
    def clean(self, properties: dict):
        pass
