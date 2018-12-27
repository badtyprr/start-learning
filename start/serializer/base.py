# Base classes for Serializable object

from pathlib import Path
from abc import ABC, abstractmethod

class Serializer(ABC):
    @abstractmethod
    def marshall(self, filepath: Path):
        pass

    @abstractmethod
    def unmarshall(self, filepath: Path):
        pass

