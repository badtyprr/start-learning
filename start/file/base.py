# Base class for all writers

from abc import ABC

class Writer(ABC):
    def __init__(self):
        pass
        #TODO: define a base writer class and also a TensorBoard mixin? and maybe also a HDF5 model dumper?