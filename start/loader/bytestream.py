# Various input and output handlers for different data types
# The base class is modeled after the Java implementation of ByteStream and derived classes.

# Python Packages
from abc import ABC, abstractmethod
from ..utils import deferrableabstractmethod
from typing import ByteString
# 3rd Party Packages
import numpy as np


class ByteStream(ABC):
    @deferrableabstractmethod
    def close(self) -> bool:
        """
        Closes the stream and releases system resources
        :return: bool type representing True if successful
        """
        pass


class InputStream(ByteStream):
    @abstractmethod
    def available(self) -> int:
        """
        Estimates the number of bytes available to read.
        :return: int type representing the best estimate of the number of bytes available to read
        """
        pass

    @abstractmethod
    def read(self, buffer: ByteString, bytes: int, offset: int=0):
        """
        Reads the specified number of bytes into buffer.
        :param buffer: bytearray type representing the byte buffer
        :param bytes: int type representing the number of bytes to read
        :param offset: int type representing the number of bytes to offset the read
        """
        pass


class OutputStream(ByteStream):
    @abstractmethod
    def flush(self) -> int:
        """
        Flushes any buffered bytes to the stream.
        :return int type representing the number of bytes written
        """
        pass

    @abstractmethod
    def write(self, buffer: ByteString, bytes: int=0, offset: int=0) -> int:
        """
        Writes the specified number of bytes to the stream.
        :param buffer: bytearray type representing the buffer to write
        :param bytes:  int type representing the number of bytes to write. '0' means all bytes.
        :param offset: int type representing the number of bytes to offset the destination
        :return: int type representing the number of bytes written
        """
        pass


class MemoryInputStream(InputStream):
    def available(self) -> int:
        """
        Estimates the number of bytes available to read.
        :return: int type representing the best estimate of the number of bytes available to read
        """
        return 0

    def read(self, buffer: ByteString, bytes: int, offset: int = 0):
        """
        Reads the specified number of bytes into buffer.
        :param buffer: bytearray type representing the byte buffer
        :param bytes: int type representing the number of bytes to read
        :param offset: int type representing the number of bytes to offset the read
        """
        pass


class MemoryOutputStream(OutputStream):
    def flush(self) -> int:
        """
        Flushes any buffered bytes to the stream.
        :return int type representing the number of bytes written
        """
        return 0

    def write(self, buffer: ByteString, bytes: int = 0, offset: int = 0) -> int:
        """
        Writes the specified number of bytes to the stream.
        :param buffer: bytearray type representing the buffer to write
        :param bytes:  int type representing the number of bytes to write. '0' means all bytes.
        :param offset: int type representing the number of bytes to offset the destination
        :return: int type representing the number of bytes written
        """
        return 0


class FileInputStream(InputStream):
    def available(self) -> int:
        """
        Estimates the number of bytes available to read.
        :return: int type representing the best estimate of the number of bytes available to read
        """
        return 0

    def read(self, buffer: ByteString, bytes: int, offset: int = 0):
        """
        Reads the specified number of bytes into buffer.
        :param buffer: bytearray type representing the byte buffer
        :param bytes: int type representing the number of bytes to read
        :param offset: int type representing the number of bytes to offset the read
        """
        pass


class FileOutputStream(OutputStream):
    def flush(self) -> int:
        """
        Flushes any buffered bytes to the stream.
        :return int type representing the number of bytes written
        """
        return 0

    def write(self, buffer: ByteString, bytes: int = 0, offset: int = 0) -> int:
        """
        Writes the specified number of bytes to the stream.
        :param buffer: bytearray type representing the buffer to write
        :param bytes:  int type representing the number of bytes to write. '0' means all bytes.
        :param offset: int type representing the number of bytes to offset the destination
        :return: int type representing the number of bytes written
        """
        return 0


class HDF5OutputStream(FileOutputStream):
    pass

