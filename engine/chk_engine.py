import os
import time
from abc import ABCMeta, abstractmethod
from datetime import timedelta
from multiprocessing import Process
from typing import Dict, List, Optional

from MujicaChk.engine.shmengine import (
    MUJICA_CKPT_CONFIG_KEY,
    SharedMemoryEngine,
    CheckpointConfig,
    SharedMemoryObjectPrefix
)

from MujicaChk.utils.env_utils import(
    get_local_rank,
    get_group_rank,
)
from MujicaChk.utils.time_utils import (
    cuda_timer,
    timer
)


from multiprocessing import shared_memory
from MujicaChk.common.constants import CheckpointConstant,CheckpointMetaKey
from MujicaChk.utils.chk_utils import TensorMeta
from MujicaChk.utils.log import default_logger as log
from MujicaChk.utils import env_utils

import torch
import torch.distributed as dist

from MujicaChk.common.singleton import Singleton

def _local_rank0_log(local_rank, message):
    if local_rank == 0:
        log.info(message)

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
        # TODO this part used to Asynchronous writes. We are not needed for the time being
        # if not self.save_proc:
        #     self.save_proc = start_saver_process()

        self.checkpoint_dir = checkpoint_dir
        self._save_timeout = save_timeout
        
        self._cached_step = 0 # init step = 0 

        self._local_rank = env_utils.get_local_rank() # IMPORTANT 本地rank
        self._world_size = 1
        self._local_size = 1
        # 
       
        self._saving_ranks: Optional[List[int]] = None
        self._saver_group = None
        self._loader_group = None
        self._init_sync_group(comm_backend)

        self._shm_handler = SharedMemoryEngine(self._local_rank, host=(self._local_rank == 0))
        """
        My basic idea is to build a distributed checkpoint system based on DeepSpeed. 
        The content we need to save under the ZeRO architecture is designed as follows
        In the version 1 system, we mainly target ZeRO-1,ZeRO-2

        Node1                           | Node2         
        G0      G1      G2      G3        G4      G5      G6      G7
        M                                 M
        O1      O2      O3      O4        O1      O2      O3      O4

        M:model
        O[i]: partition i of optimizer

        Obviously, we don't need each GPU to store its model parameters, and we don't 
        even need to store model parameters (as long as at least one machine survives). 
        In our design, we need to create a SharedMemoryEngine for each GPU.
        """

    def _init_sync_group(self, comm_backend):
        if not dist.is_initialized():
            self._saving_ranks = [0]
            return

        self._rank = dist.get_rank() # 全局rank
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
        # self._saving_ranks = self.get_saving_ranks()
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
            
    
    # def __del__(self):
    #     self.close()

    def close(self):
        """Close the shared memory."""
        # if self._shm_handler:
        #     try:
        #         self._shm_handler.close()
        #         self._shm_handler = None
        #     except OSError as e:
        #         print(f"Error closing shared memory: {e}")
        self._shm_handler.close()

    def save_state_dict_to_memory(self, state_dict, conf:CheckpointConfig):
        # if self._local_rank != self.local_shared_id:
        #     return False
        # if self._saving_ranks and self._rank not in self._saving_ranks:
        #     return False

        conf.rank = self._rank
        conf.group_rank = self._group_rank
        conf.world_size = self._world_size
        #all_rank_ready = check_all_rank_ready(self._saver_group, is_ready = True)
        state_dict[MUJICA_CKPT_CONFIG_KEY] = conf
        self._shm_handler.save_state_dict(state_dict)
        self._cached_step = conf.step
        #print(f"save_dict_print test\n",state_dict)
        return True

    def get_state_dict_from_memory(self, read_meta_dict):
        state_dict = {}
        # default_config = CheckpointConfig()
        # config = self._shm_handler.get_checkpoint_config(default_config)
        # Get optimizer state_dict
        # if _model_FLAG == False or self._local_rank == 0:
        #     state_dict = self._shm_handler.load_state_dict(read_meta_dict)
        # elif _model_FLAG == True and self._local_rank != 0:
        #     self._shm_handler_model_rank_0 =  SharedMemoryEngine(local_rank=0)
        #     state_dict = self._shm_handler_model_rank_0.load_state_dict(read_meta_dict)
        #     self._shm_handler_model_rank_0.close()
        #state_dict = self._shm_handler.load_state_dict(read_meta_dict)
        # Read_key = next(iter(read_meta_dict))
        # print(F"!!!rEAD _KEY {Read_key}")
        # if Read_key == CheckpointMetaKey.OPTIMIZER_STATE_DICT or self._local_rank == 0:
        #     state_dict = self._shm_handler.load_state_dict(read_meta_dict)
        # if Read_key == CheckpointMetaKey.MODEL and self._local_rank != 0:
        #     modulename = SharedMemoryObjectPrefix.SHM_NAME + str(0)
        #     # shm = SharedMemory(name=modulename)
        #     # state_dict = _traverse_read_dict_from_shm(read_meta_dict, shm)
        #     # shm.close()
        #     shm = shared_memory(name = modulename)
        #     state_dict = _traverse_read_dict_from_shm(read_meta_dict, shm)
        #     shm.close()
        state_dict = self._shm_handler.load_state_dict(read_meta_dict)
        state_dict.pop(MUJICA_CKPT_CONFIG_KEY, None)
        return state_dict
    
    # @abstractmethod
    # def get_saving_ranks(self):
    #     pass