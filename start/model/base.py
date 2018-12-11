# Base class for neural networks

from abc import ABC, abstractmethod

class NeuralNetwork(ABC):
    def __init__(self):
        super().__init__()

    @staticmethod
    @abstractmethod
    def build(properties: dict):
        pass
