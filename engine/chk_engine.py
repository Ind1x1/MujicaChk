import os
import time
from abc import ABCMeta, abstractmethod
from datetime import timedelta
from multiprocessing import Process
from typing import Dict, List, Optional

from engine.shmengine import (
    MUJICA_CKPT_CONFIG_KEY,
    SharedMemoryEngine,
    CheckpointConfig,
    SharedMemoryObjectPrefix
)

from utils.env_utils import(
    get_local_rank,
    get_group_rank,
)
from utils.time_utils import (
    cuda_timer,
    timer
)

from common.constants import CheckpointConstant

from utils.chk_utils import TensorMeta
from utils.rank_logger import RankLogger as log
from utils import env_utils

import torch
import torch.distributed as dist

from common.singleton import Singleton

def _local_rank0_log(local_rank, message):
    if local_rank == 0:
        logger.info(message)

class ReadyTensor(Singleton):
    def __init__(self, device) -> None:
        self.tensor = torch.tensor([0], dtype=torch.int32).to(device)

def check_all_rank_ready(group: dist.ProcessGroup, is_ready: bool) -> bool:
    """
    Determine if all ranks are ready.
    """
    if group is None or not dist.is_initialized():
        return is_ready
    backend = dist.get_backend(group)
    rank = dist.get_rank()
    device = device = "cpu" if backend == "gloo" else f"cuda:{rank}"
    rf = ReadyTensor.singleton_instance(device)
    rf.tensor[0] = 0 if is_ready else 1
    dist.all_reduce(rf.tensor, op=dist.ReduceOp.SUM, group=group)
    return rf.tensor.item() == 0

def verify_all_rank_step_consistent(group: dist.ProcessGroup, step):
    """
    Verify whether the step in all ranks are consistent.
    """
    if not group and not dist.is_initialized():
        return True
    backend = dist.get_backend(group)
    local_rank = get_local_rank()
    device = "cpu" if backend == "gloo" else f"cuda:{local_rank}"
    t = torch.tensor([float(step)]).to(device)
    if group:
        world_size = group.size()
    else:
        world_size = dist.get_world_size()
    outputs = [torch.tensor([0.0]).to(device) for _ in range(world_size)]
    dist.all_gather(outputs, t, group=group)
    succeed = True
    for step in outputs:
        if not torch.equal(step, outputs[0]):
            succeed = False
    del t, outputs
    return succeed

#TODO IMPOTANT FUNCTION
def start_saver_process():
    pass

class CheckpointEngine(metaclass=ABCMeta):
    """
    The checkpoint engine achieves low-overhead checkpoint writing by creating 
    a checkpoint process, separating the checkpoint writing operation from the 
    main training. The issue here is the timing of holding and destroying the 
    checkpoint process. In my experiment with miniGPT training, this approach 
    can control the overhead at around 10% under high frequency (close to per-turn 
    write operations). The mathematical model here lies in how to find the 
    optimal checkpoint frequency.

    Args:
        checkpoint_dir (str): the directory to save checkpoint.
        storage: a CheckpointStorage instance to write/read the storage.
        comm_backend (str): the communcation backend to create a process group,
            The default is the backend of general main process group.
    """
    
    save_proc = None

    def __init__(
            self,
            checkpoint_dir: str,
            # storage: CheckpointStorage,
            comm_backend: str= "",
            save_timeout: int = CheckpointConstant.SAVE_TIMEOUT
    ):
        if not self.save_proc:
            self.save_proc = start_saver_process()

        self.checkpoint_dir = checkpoint_dir

        self._save_timeout = save_timeout
        self._local_rank = 0

        self._cached_step = 0

        self._world_size = 1
        self._group_rank = 1
        self._local_size = 1

        # 
        self._shm_handler = SharedMemoryEngine(self._local_size, host=False)
        self._saving_ranks: Optional[List[int]] = None
        self._saver_group = None
        self._loader_group = None

    def _init_sync_group(self, comm_backend):
        if not dist.is_initialized():
            self._saving_ranks = [0]
            return

        self._rank = dist.get_rank()
        self._group_rank = env_utils.get_group_rank()
        self._world_size = dist.get_world_size()
        backend = comm_backend if comm_backend else dist.get_backend()
        if backend == dist.get_backend():
            self._loader_group = None
        else:
            self._loader_group = dist.new_group(
                backend=backend,
                timeout=timedelta(seconds=60),
            )
        self._saving_ranks = self.get_saving_ranks()
        if backend == dist.get_backend() and (
            self._saving_ranks is None
            or len(self._saving_ranks) == dist.get_world_size()
        ):
            self._saver_group = None
            message = (
                "Use the default process group to sync "
                "when saving checkpoint."
            )
            _local_rank0_log(self._local_rank, message)
        else:
            self._saver_group = dist.new_group(
                ranks=self._saving_ranks,
                backend=backend,
                timeout=timedelta(seconds=60),
            )
            if self._saving_ranks:
                message = (
                    f"Create a {backend} commumication group to save "
                    f"checkpoint. Saving ranks are {self._saving_ranks}."
                )
            else:
                message = (
                    f"Create a {backend} commumication group to save "
                    "checkpoint. Saving ranks are all ranks."
                )
            _local_rank0_log(self._local_rank, message)
    
    def __del__(self):
        self.close()

    def close(self):
        """Close the shared memory."""
        self._shm_handler.close()

    def save_state_dict_to_memory(self, state_dict, conf:CheckpointConfig):
        if self._local_rank != self.local_shard_id:
            return False
        if self._saving_ranks and self._rank not in self._saving_ranks:
            return False
        
        conf.rank = self._rank
        conf.group_rank = self._group_rank
        conf.world_size = self._world_size

        all_rank_ready = check_all_rank_ready(self._saver_group, is_ready = True)
        self._shm_handler.save_state_dict(state_dict)
        state_dict[MUJICA_CKPT_CONFIG_KEY] = conf
        self._cached_step = conf.step

    def get_state_dict_from_memory(self):
        pass