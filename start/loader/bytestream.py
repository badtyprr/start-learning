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

    @property
    def file(self):
        return self._file

    @file.setter
    def file(self, f):
        _validate_types(f=Union[file_t, path_t])
        self._file = f

    def close(self) -> bool:
        """
        Closes the stream and releases system resources
        :return: bool type representing True if successful
        """
        # Try to close a file descriptor
        try:
            self._file.close()
        except AttributeError:
            # Not a file descriptor
            return False
        # Closed file descriptor
        return True

    def available(self) -> int:
        """
        Estimates the number of bytes available to read.
        :return: int type representing the best estimate of the number of bytes available to read
        """
        try:
            return os.stat(self._file).st_size
        except FileNotFoundError:
            # Try file descriptor
            marker = self._file.tell()
            available_bytes = self._file.seek(0, os.SEEK_END)
            self._file.seek(marker)
            return available_bytes
        except AttributeError:
            raise TypeError('_file is neither a path, nor a file descriptor')

    def read(self, buffer: ByteString, bytes: int, offset: int = 0):
        """
        Reads the specified number of bytes into buffer.
        :param buffer: bytearray type representing the byte buffer
        :param bytes: int type representing the number of bytes to read
        :param offset: int type representing the number of bytes to offset the read
        """
        # Don't write past the bytearray buffer, although Python will technically do it
        bytes = min(len(buffer), bytes)
        try:
            buffer[:] = self._file.read(bytes)
        except AttributeError:
            # Not a file descriptor, try to open the path
            fd = open(self._file, 'rb')
            buffer[:] = fd.read(bytes)
            fd.close()
        except FileNotFoundError:
            raise TypeError('_file is neither a path, nor a file descriptor')


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

