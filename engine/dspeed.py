import os
from typing import Dict

import torch
import torch.distributed as dist

from deepspeed.runtime.engine import DeepSpeedEngine
from deepspeed.runtime.zero.config import ZeroStageEnum

from MujicaChk.engine.checkpointer import Checkpointer
from MujicaChk.engine.dspeed_engine import DeepSpeedCheckpointEngine

from MujicaChk.utils import env_utils
from MujicaChk.common.constants import CheckpointConstant


torch_native_save = torch.save
torch_native_load = torch.load

class DeepSpeedCheckpointer(Checkpointer):
    """
    MujicaChk checkpointer saves and load model

    Examples::
        >>> model, optimizer, _, lr_scheduler = deepspeed.initialize(...)
        >>> MujicaCheckpointer = DeepSpeedCheckpointer(engine, save_dir) 
        >>> if args.save_model_step is not None and global_step % args.save_model_step == 0:
        >>>     MujicaCheckpointer.save_checkpoint(tag)

    Version1.0 we test in ZeRO-1 and ZeRO-2    
    """
    def __init__(
        self,
        engine: DeepSpeedEngine,
        checkpoint_dir,
        comm_backend = "",
        #deletion_strategy=None,
        save_timeout = CheckpointConstant.SAVE_TIMEOUT,
    ):
        self.engine = engine
        self.checkpoint_dir = checkpoint_dir
        
        global_shard_num = 1
        if self.engine.zero_optimization():
            global_shard_num = dist.get_world_size(
                self.engine.optimizer.dp_process_group
            )
        zero_stage = self.engine.zero_optimization_stage()
        self._local_rank = env_utils.get_local_rank()

        if zero_stage < ZeroStageEnum.weights and self._local_rank == 0:
            self.engine.save_non_zero_checkpoint = True

        self.dscheckpointengine = DeepSpeedCheckpointEngine(
            checkpoint_dir,
            global_shard_num = global_shard_num,
            zero_stage = zero_stage,
            comm_backend = comm_backend,
            save_timeout = save_timeout
        )

    """
    *********save part***********
    """
    def save_checkpoint(
        self, 
        save_dir,
        tag = None,
        client_state = {},
        save_latest = True,
    ):
        self._save_shm_checkpoint(
                save_dir, tag, client_state, save_latest
            )
        
    def _save_shm_checkpoint(
        self, save_dir, tag=None, client_state={}, save_latest=True    
    ):
        torch.save = self.dscheckpointengine._save_state_dict
        self.engine.save_checkpoint(save_dir, tag, client_state, save_latest)
        torch.save = torch_native_save
        self.dscheckpointengine.save_to_memory(
            self.engine.global_steps,
            self.dscheckpointengine.state_dict,
            self.dscheckpointengine.paths,
        )
        #self._update_tracer_file(tag)
    """
    *********load part***********
    """
    def load_checkpoint(
        self,
        load_dir,
        tag=None,
        load_module_strict=True,
        load_optimizer_states=True,
        load_lr_scheduler_states=True,
        load_module_only=False,
        custom_load_fn=None,
    ):
        """
        Load a checkpointing state dict

        Args:
            the same as the DeepSpeed Engine.LOAD_CHECKPOINT
        """
        