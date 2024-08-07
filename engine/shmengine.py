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

from utils.env_utils import(
    get_local_rank,
    get_group_rank,
)
from common.multi_process import(
    SharedMemory,
    CheckpointDict,
)

from utils.chk_utils import(
    TensorMeta,
    _read_shared_memory,
    _traverse_read_dict_from_shm,
    _traverse_state_dict,
    _traverse_copy_to_shm,
    _write_shared_memory,
    _create_shared_memory
)
import torch

MUJICA_CKPT_CONFIG_KEY = "MUJICA_CKPT_CONFIG"

class SharedMemoryObjectPrefix:
    META_NAME = "checkpoint_meta_"
    SHM_NAME = "checkpoint_shm_"

@dataclass
class CheckpointConfig:
    """
    The configuration of a checkpointing shard on the training process.

    Attributes:
        step (int): the global interation step.
        writing_shm (bool): the flag whether the training process is writing
            the state dict into the shared memory.
        paths (dict): the key is in ["model_state", "optim_state"] and the
            value is path.
    """

    rank: int = 0
    group_rank: int = 0
    world_size: int = 0
    step: int = 0
    writing_shm: bool = False
    paths: Dict[str, str] = None  # type: ignore

class SharedMemoryEngine(object):
    
    def __init__(self, local_rank, host=True):
        self._buffer_size = 0 # BUFFER STAR
        meta_name = SharedMemoryObjectPrefix.META_NAME + str(local_rank)
        # NEED OPTIMIZER
        self._shm_name = SharedMemoryObjectPrefix.SHM_NAME + str(local_rank)
        self.shared_memory: Optional[SharedMemory] = None
        self.meta_dict = None
        #TODO
        #self.metadata = CheckpointDict(name=meta_name)
        self._creation_FLAG = True

    def close(self):
        if self.shared_memory:
            self.shared_memory.close()

    def unlink(self):
        if not self.shared_memory:
            # From dlrover:
            # The shared memory may be created by other processes.
            self.init_shared_memory()
        if self.shared_memory:
            self.shared_memory.unlink()
        # if self.metadata:
        #     self.metadata.unlink()
    
    def reset(self):
        self._creation_FLAG = True

    def _create_tensor_meta(self, value: torch.Tensor):
        if not torch.is_tensor(value):
            return value
        meta = TensorMeta(
            shape=tuple(value.shape),
            dtype=value.dtype,
            element_size=value.element_size(),
            numel=value.numel(),
            offset=self._buffer_size,
        )
        self._buffer_size += value.numel() * value.element_size()
        return meta
    """
    layer1: tensor([[ 0.1737,  1.5381,  0.5881],
                    [ 0.1285, -0.2957, -1.9910],
                    [ 1.1275, -2.1310, -0.4366]])
    state: training
    optimizer: {'lr': 0.001, 'momentum': 0.9}
    data: [tensor([[-0.1398, -0.4347],
                   [-1.2835, -2.4825]]), 
           tensor([[-0.8753,  0.9444, -0.6177],
                   [-0.0344,  0.6343, -0.2200]])]

    meta_dict:
    layer1: TensorMeta(shape=(3, 3), dtype=torch.float32, element_size=4, numel=9, offset=0)
    state: training
    optimizer: {'lr': 0.001, 'momentum': 0.9}
    data:  [TensorMeta(shape=(2, 2), dtype=torch.float32, element_size=4, numel=4, offset=36), 
            TensorMeta(shape=(2, 3), dtype=torch.float32, element_size=4, numel=6, offset=52)]
    """
    def init_meta_dict(self,state_dict):
        self.meta_dict = _traverse_state_dict(state_dict, self._create_tensor_meta)
        self.meta_dict[MUJICA_CKPT_CONFIG_KEY] = CheckpointConfig()
        return

    def init_shared_memory(self, create=False, size=0):
        self.shared_memory = _create_shared_memory(
            self._shm_name, create=create, size=size
        )
        self._creation_FLAG = False

    def save_state_dict(self, state_dict):
        if not self.shared_memory:
            # meta_dict = _traverse_state_dict(
            #     state_dict, self._create_tensor_meta
            # )
            self.init_meta_dict(state_dict)
            self.init_shared_memory(create=True, size=self._buffer_size)

        # get existing meta_dict
        # else:
        #     meta_dict = self.metadata.get(local=True)
        ckpt_conf: CheckpointConfig = self.meta_dict[MUJICA_CKPT_CONFIG_KEY]
        ckpt_conf.writing_shm = True

        # self.metadata.set(meta_dict)
        assert self.shared_memory is not None
        _traverse_copy_to_shm(state_dict, self.meta_dict, self.shared_memory.buf)
        ckpt_conf.writing_shm = False
        # self.metadata.set(meta_dict)
    
    def load_state_dict(self):
        """
        Load the state dict from the shared memory.

        Returns:
            Tuple(int, dict): The first value is the iteration step,
                the second value is the state dict.
        """
        # meta_dict = self.metadata.get()
        meta_dict = self.meta_dict
        default_config = CheckpointConfig()
        config = meta_dict.get(MUJICA_CKPT_CONFIG_KEY, default_config)
        if not meta_dict or config.writing_shm:
            return {}
        if self.shared_memory is None or self._creation_FLAG:
            self.init_shared_memory(create=False)
        if not self.shared_memory:
            return {}
        
        state_dict = _traverse_read_dict_from_shm(meta_dict, self.shared_memory)
        return state_dict
    
    # def no_checkpint_state(self):
    #     """
    #     The handler lazily initializes the shared memory. The shared memory
    #     of the handler on the host may be None even if the handler on the
    #     device has saved state dict.
    #     """
    #     #meta_dict = self.metadata.get()
    #     config: CheckpointConfig = meta_dict.get(MUJICA_CKPT_CONFIG_KEY, None)
    #     if config is None or config.step == 0:
    #         return True
    #     return False

    # def get_checkpoint_config(self, default_config):
    #     """
    #     Get the configuration of checkpointing state dict in the shared
    #     memory.

    #     Returns:
    #         A CheckpointShardConfig instance.
    #     """
    #     meta_dict = self.metadata.get()
    #     config = meta_dict.get(MUJICA_CKPT_CONFIG_KEY, default_config)
    #     return config