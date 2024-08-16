import copy
from typing import Dict
from utils import env_utils
from .chk_engine import CheckpointEngine
from common.constants import CheckpointConstant
from .checkpointer import Checkpointer

from engine.shmengine import (
    MUJICA_CKPT_CONFIG_KEY,
    SharedMemoryEngine,
    CheckpointConfig,
    SharedMemoryObjectPrefix
)

from utils.time_utils import (
    cuda_timer,
    timer
)
_DS_MODEL_SD_FILE_SUFFIX = "model_states.pt"
_DS_OPTIM_SD_FILE_SUFFIX = "optim_states.pt"

import torch
import torch.distributed as dist

torch_native_save = torch.save
torch_native_load = torch.load

class DeepSpeedCheckpointEngine(CheckpointEngine):
    """
    The checkpoint engine synchronously writes the state dict of 
    `DeepSpeedEngine` into the shared memory and notify the agent
    in main process to asynchronously save the state dict from the shared
    memory into the storage.

    Attributes:
        checkpoint_dir (str):  the directory to save the temp checkpoint
            if the training process fails.
        dp_size (int): the world size of data parallelism.
        global_shard_num (int): the number of shards across all ranks.
        zero_stage (int): the DeepSpeed ZERO Stage number.
        comm_backend (str): the backend to synchronize when saving the
            checkpoint to the memory.
    """
    def __init__(
        self,
        checkpoint_dir,
        global_shard_num = 1,
        zero_stage = 0,
        comm_backend = "",
        save_timeout = CheckpointConstant.SAVE_TIMEOUT,  
    ):
        self.state_dict: Dict[str, object] = {}
        self.paths: Dict[str, str] = {}
        self.global_shard_num = global_shard_num
        self.zero_stage = zero_stage
        super().__init__(checkpoint_dir, comm_backend, save_timeout)

    def get_saving_ranks(self):
        """
        Get the ranks which need to save the sharding state dict into
        the memory.
        """
        world_size = dist.get_world_size()
        local_world_size = env_utils.get_local_world_size()
        save_ranks = []
        local_shard_num = self.get_local_shard_num()
        for i in range(world_size):
            local_rank = i % local_world_size
            if local_rank < local_shard_num:
                save_ranks.append(i)
        return save_ranks
    
    def _save_state_dict(self, state_dict, path: str):
        """
        state_dict: 
            model_state: model_state:[Dict]
            optimizer_state: optimizer_state[Dict]

        path:
            model_state: ./outputtest/global_step1/mp_rank_00_model_states.pt
            optimizer_state: 

        We use this function to build the state_dict we want to save
        """
        if not isinstance(path, str):
            torch_native_save(state_dict, path)
            return
        if path.endswith(_DS_MODEL_SD_FILE_SUFFIX):
            sd_name = CheckpointConstant.MODEL_STATES_NAME
        elif path.endswith(_DS_OPTIM_SD_FILE_SUFFIX):
            sd_name = CheckpointConstant.OPTIM_STATES_NAME
        else:
            sd_name = path.split("/")[-1]
        if sd_name:
            self.state_dict[sd_name] = state_dict
            self.paths[sd_name] = path

    @timer
    def save_to_memory(self, step, state_dict, paths):
        conf = CheckpointConfig(step=step, paths=paths)
        success = self.save_state_dict_to_memory(state_dict, conf)
        return success