# Pandas mixin for handling Pandas dataframes

# Python Packages
from abc import ABC, abstractmethod
# 3rd Party Packages
import pandas as pd
# User Packages
from .base import Dataset

class CSVDatasetMixin(Dataset, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Flag using Pandas dataframe
        self._pandas = True
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
        if not properties:
            properties = {}
        self.dataframe = pd.read_csv(self.dataset_path)
        # The inheriting class will need to implement the actual data load

    @property
    def dataframe(self) -> pd.DataFrame:
        return self.dataframe

    @dataframe.setter
    def dataframe(self, df: pd.DataFrame):
        if isinstance(df, pd.DataFrame):
            self.dataframe = df

