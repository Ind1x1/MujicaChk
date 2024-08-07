import mmap
import os
import queue
import shutil
import socket
import threading
import time

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from multiprocessing import shared_memory
from typing import Dict

import _posixshmem

DEFAULT_TMP_DIR = "/tmp/mujicachk_sock/"

SUCCESS_CODE = "OK"
ERROR_CODE = "ERROR"

class LocalSocketComm(metaclass=ABCMeta):
    """
    Local socket for processes to communicate.

    Args:
        name (str): the instance name which must be unique if multiple
            processes share a common object using the local socket.
        create (bool): If true, the instance creates a socket server;
            otherwise, the instance creates a socket client to access
            the shared object.
    """
    pass

class CheckpointDict(metaclass=ABCMeta):
    pass


class SharedMemory(shared_memory.SharedMemory):
    """
    Customize SharedMemory to prevent the default behavior of Python's 
    ResourceTracker from automatically unlinking and deleting files. In 
    case of training failure, we must ensure that new processes can 
    access the checkpoint files in shared memory to restart the training. 

    Note:: This customized SharedMemory is based on the work done by Dlrover.

    Note:: We must explicitly unlink the SharedMemory to avoid memory leak.
    """
    def __init__(self, name=None, create=False, size=0):
        self._name = None
        self._fd = 1
        self._mmap = None
        self._flags = os.O_RDWR
        self._mode = 0o600
        self._prepend_leading_slash = True

        if not size >= 0:
            raise ValueError("'size' must be a positive integer")
        if create:
            self._flags = os.O_CREAT | os.O_EXCL | os.O_RDWR
            if size == 0:
                raise ValueError(
                    "'size' must be a positive number different from zero"
                )
        if name is None and not self._flags & os.O_EXCL:
            raise ValueError("'name' can only be None if create=True")
        if name is None:
            while True:
                name = shared_memory._make_filename()
                try:
                    self._fd = _posixshmem.shm_open(
                        name, self._flags, mode=self._mode
                    )
                except FileExistsError:
                    continue
                self._name = name
                break
        else:
            name = "/" + name if self._prepend_leading_slash else name
            self._fd = _posixshmem.shm_open(name, self._flags, mode=self._mode)
            self._name = name
        try:
            if create and size:
                os.ftruncate(self._fd, size)
            stats = os.fstat(self._fd)
            size = stats.st_size
            self._mmap = mmap.mmap(self._fd, size)
        except OSError:
            self.unlink()
            raise

        self._size = size
        self._buf = memoryview(self._mmap)

    def unlink(self):
        """Requests that the underlying shared memory block be destroyed.

        In order to ensure proper cleanup of resources, unlink should be
        called once (and only once) across all processes which have access
        to the shared memory block.
        """
        if self._name:
            try:
                _posixshmem.shm_unlink(self._name)
            except FileNotFoundError:
                pass
    
