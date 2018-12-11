# Pandas mixin for handling Pandas dataframes

import pandas as pd

class PandasDatasetMixin(object):
    def __init__(self):
        self._pandas = True

    def load_dataframe(self, dataframe_path):
        if dataframe_path:
            self.dataframe = pd.read_csv(dataframe_path)

    @property
    def dataframe(self) -> pd.DataFrame:
        return self.dataframe

    @dataframe.setter
    def dataframe(self, df: pd.DataFrame):
        if isinstance(df, pd.DataFrame):
            self.dataframe = df

