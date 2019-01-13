# Various input and output handlers for different data types
# The base class is modeled after the Java implementation of ByteStream and derived classes.

# Python Packages
from abc import ABC, abstractmethod
from typing import ByteString, Union
from pathlib import Path
from io import IOBase
import os
# 3rd Party Packages
import numpy as np
# User Packages
from ..utils import deferrableabstractmethod, _validate_types

# Types
file_t = IOBase
path_t = Union[str, Path]


class ByteStream(ABC): # abstract
    @deferrableabstractmethod
    def close(self) -> bool:
        """
        Closes the stream and releases system resources
        :return: bool type representing True if successful
        """
        pass


class InputStream(ByteStream): # abstract
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


class OutputStream(ByteStream): # abstract
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
    def close(self) -> bool:
        """
        Closes the stream and releases system resources
        :return: bool type representing True if successful
        """
        return True

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
    def close(self) -> bool:
        """
        Closes the stream and releases system resources
        :return: bool type representing True if successful
        """
        return True

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
    def __init__(self, file: Union[file_t, path_t]):
        """
        FileInputStream returns the bytearray contents of a file
        :param file: type Union[IOBase, str, Path] representing a file pointer, or path to the file, respectively
        """
        self._file = file
        # Convert a path to a file descriptor, if necessary
        self._to_file_descriptor()

    @property
    def file(self):
        return self._file

    @file.setter
    def file(self, f):
        _validate_types(f=Union[file_t, path_t])
        self._file = f
        # Convert a path to a file descriptor, if necessary
        self._to_file_descriptor()

    @classmethod
    def _to_file_descriptor(cls):
        try:
            # Do nothing if it's already a descriptor
            _validate_types(cls._file, IOBase)
        except TypeError:
            # Open the (possible) path
            try:
                cls._file = open(cls._file, 'rb')
            except FileNotFoundError:
                raise TypeError('_file is neither a path, nor a file descriptor')

    @classmethod
    def _reopen_file_descriptor(cls):
        print('WARNING: file was closed before access, reopening...')
        cls._file = open(cls._file.name, cls._file.mode)

    def close(self) -> bool:
        """
        Closes the stream and releases system resources
        :return: bool type representing True if successful
        """
        self._file.close()
        return True

    def available(self) -> int:
        """
        Estimates the number of bytes available to read.
        :return: int type representing the best estimate of the number of bytes available to read
        """
        # Try file descriptor
        try:
            marker = self._file.tell()
        except ValueError:
            # If the file descriptor was closed, open it again
            marker = 0
            self._reopen_file_descriptor()
        available_bytes = self._file.seek(0, os.SEEK_END)
        self._file.seek(marker)
        return available_bytes

    def read(self, buffer: ByteString, bytes: int, offset: int = 0):
        """
        Reads the specified number of bytes into buffer.
        :param buffer: bytearray type representing the byte buffer
        :param bytes: int type representing the number of bytes to read
        :param offset: int type representing the number of bytes to offset the read
        """
        # Don't write past the bytearray buffer, although Python will technically do it
        bytes = min(len(buffer), bytes, self.available())
        try:
            buffer[:] = self._file.read(bytes)
        except ValueError:
            # Try to reopen the file descriptor
            self._reopen_file_descriptor()
            buffer[:] = self._file.read(bytes)


class FileOutputStream(OutputStream):
    def close(self) -> bool:
        """
        Closes the stream and releases system resources
        :return: bool type representing True if successful
        """
        return True

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

