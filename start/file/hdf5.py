# HDF5 handling for Tensorflow models

# 3rd Party Packages
import h5py
# User Packages
from .file import DatasetFileWriter


class HDF5DatasetFileWriter(DatasetFileWriter):
    def __init__(self, data_key: str='images', buffer_size=1000):
        super().__init__()
        # Open the HDF5 databases
        self.db = h5py.File(self.filepath, 'w')
        self.data = self.db.create_dataset(
            data_key, self.dimensions, dtype='float')
        self.labels = self.db.create_dataset(
            'labels', (self.dimensions[0],), dtype='int')

        # Initialize buffer
        self.buffer_size = buffer_size
        self.buffer = {'data': [], 'labels': []}
        self.idx = 0

    def add(self, rows, labels):
        # Add data and labels
        # Extend will add elements instead of a list of elements like append does
        self.buffer['data'].extend(rows)
        self.buffer['labels'].extend(labels)
        # Check if buffer exceeded
        if len(self.buffer['data']) >= self.buffer_size:
            self.flush()

    def flush(self):
        # Write buffer to disk
        idx_end = self.idx + len(self.buffer['data'])
        self.data[self.idx:idx_end] = self.buffer['data']
        self.idx = idx_end
        self.buffer = {'data': [], 'labels': []}

    def flush_class_labels(self, class_labels):
        label_set = self.db.create_dataset(
            'label_names',
            (len(class_labels),),
            dtype=h5py.special_dtype(vlen=str)
        )
        label_set[:] = class_labels

    def close(self):
        # Flush leftover buffer
        if len(self.buffer['data']) > 0:
            self.flush()
        # Close file
        self.db.close()

