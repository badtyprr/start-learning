# Pandas mixin for handling Pandas dataframes

# 3rd Party Packages
import pandas as pd
# User Packages
from .base import Dataset

class CSVDatasetMixin(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Flag using Pandas dataframe
        self._pandas = True
        # Add handler for files
        self.handlers['csv'] = self._csv_dataframe_handler

    def _csv_dataframe_handler(self, properties: dict=None):
        if not properties:
            properties = {}
        self.dataframe = pd.read_csv(self.dataset_path)
        raise NotImplementedError('Loading images from a Pandas dataframe is not yet supported')

    @property
    def dataframe(self) -> pd.DataFrame:
        return self.dataframe

    @dataframe.setter
    def dataframe(self, df: pd.DataFrame):
        if isinstance(df, pd.DataFrame):
            self.dataframe = df

