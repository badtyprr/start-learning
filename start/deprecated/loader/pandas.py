# Pandas mixin for handling Pandas dataframes

# Python Packages
from abc import ABC, abstractmethod
import os
# 3rd Party Packages
import pandas as pd
# User Packages
from .base import Dataset

class CSVDatasetMixin(Dataset, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Flag using Pandas dataframe
        self._pandas = True
        self._dataframe = None
        # Add handler for files
        self.handlers['csv'] = self._csv_handler

    @abstractmethod
    def _csv_handler(self, properties: dict=None):
        """
        CSV file handler, just opens the dataframe.
        The inheriting class must implement the actual data load
        :param properties:
        :return:
        """
        if properties is None:
            properties = {}
        self.load_dataframe()
        # The inheriting class will need to implement the actual data load

    def load_dataframe(self):
        self.dataframe = pd.read_csv(self.dataset_path)

    @property
    def dataframe(self) -> pd.DataFrame:
        if self._dataframe is None:
            if os.path.isfile(self.dataset_path):
                filepath, ext = os.path.splitext(self.dataset_path)
                if ext[1:] == 'csv':
                    self.load_dataframe()
        return self._dataframe

    @dataframe.setter
    def dataframe(self, df: pd.DataFrame):
        if isinstance(df, pd.DataFrame):
            self._dataframe = df
        else:
            raise ValueError('dataframe must be a pandas dataframe')

