import os
import pickle
import signal
import threading
import time

from abc import ABCMeta, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Tuple

import torch

from common.multi_process import SharedMemory


@dataclass
class TensorMeta:
    shape: Tuple[int] = None
    dtype: torch.dtype = None
    element_size: int = 0
    numel: int = 0
    offset: int = 0
"""
layer1: tensor([[ 1.0587, -0.4412, -0.0078],
                [ 1.3530, -0.9772,  1.6090],
                [-0.4240,  1.4990,  1.5745]])

layer1: TensorMeta(shape=(3, 3), dtype=torch.float32, 
                   element_size=4, numel=9, offset=0)
"""

def _traverse_state_dict(value:object, visitor:Callable[[object], None]):
    """
    Invoke "visitor" for each key: value recursively in "state_dict"
    """
    if isinstance(value, Mapping):
        return {k: _traverse_state_dict(v, visitor) for k,v in value.items()}
    elif isinstance(value, List):
        return [_traverse_state_dict(v, visitor) for v in value]
    else:
        return visitor(value)

def _write_shared_memory(value: torch.Tensor, meta:TensorMeta, buffer):
    """
    Write a "CPU tensor" into the shared memory.
    """
    if value.numel() > 0:
        shm_tensor = torch.frombuffer(
            buffer, dtype=value.dtype, count=value.numel(), offset=meta.offset
        ).reshape(value.shape)
        shm_tensor.copy_(value)

def _traverse_copy_to_shm(value, meta, buffer):
    """
    Traverse the input value and copy tensors to shared memory.
    """
    if isinstance(value, Mapping):
        for k, v in value.items():
            m = meta[k]
            if isinstance(v, (Mapping, List)):
                _traverse_copy_to_shm(v, m, buffer)
            elif torch.is_tensor(v):
                _write_shared_memory(v, m, buffer)
            else:
                meta[k] = v
    elif isinstance(value, List):
        for i, v in enumerate(value):
            m = meta[i]
            if isinstance(v, (Mapping, List)):
                _traverse_copy_to_shm(v, m, buffer)
            elif torch.is_tensor(v):
                _write_shared_memory(v, m, buffer)
            else:
                meta[i] = v

def _read_shared_memory(value, shm_tensor_buffer):
    """
    Read a tensor from the buffer of shared memory.
    """
    if isinstance(value, TensorMeta):
        if value.numel == 0:
            return torch.tensor([], dtype=value.dtype)
        else:
            shm_tensor = torch.frombuffer(
                buffer=shm_tensor_buffer.buf,
                dtype=value.dtype,
                offset=value.offset,
                count=value.numel,
            )
            value = shm_tensor.reshape(value.shape)
            return value
    else:
        return value

def _traverse_read_dict_from_shm(meta_dict, tensor_shm):
    state_dict = _traverse_state_dict(
        meta_dict,
        lambda x: _read_shared_memory(x, tensor_shm),
    )
    return state_dict

def _create_shared_memory(name, create, size=0):
    """
    Create a shared memory.
    """
    if not create:
        try:
            return SharedMemory(name=name)
        except FileNotFoundError:
            return None
    if create and size == 0:
        logger.warning("Cannot create the shared memory with size = 0.")
        return None
    try:
        shm = SharedMemory(
            name=name,
            create=create,
            size=size,
        )
    except FileExistsError:
        shm = SharedMemory(name=name)
        if shm.size != size:
            logger.info(
                f"The old size is {shm.size} and "
                f"create a new memory buffer with size {size}."
            )
            shm.unlink()
            shm = SharedMemory(
                name=name,
                create=create,
                size=size,
            )
    return shm
