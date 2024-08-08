import sys
import os

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_path)

from dataclasses import dataclass
from typing import Tuple, Callable, Any, Mapping, List

from engine.shmengine import SharedMemoryEngine
from engine.chk_engine import (
    ReadyTensor,
    check_all_rank_ready,
    verify_all_rank_step_consistent
)

import torch
import torch.distributed as dist
from torch.multiprocessing import Process

def verify_all_rank_step_consistent(group: dist.ProcessGroup, step):
    """
    Verify whether the step in all ranks are consistent.
    """
    if not group and not dist.is_initialized():
        return True
    backend = dist.get_backend(group)
    local_rank = dist.get_rank()
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

def init_process(rank, size, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)
    dist.destroy_process_group()

def run(rank, size):
    step = 10  # You can change this value to test
    result = verify_all_rank_step_consistent(None, step)
    print(f"Rank {rank}: Step consistent: {result}")

if __name__ == "__main__":
    size = 4  # Number of GPUs
    processes = []
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

# def init_process(rank, size, fn, backend='nccl'):
#     """ Initialize the distributed environment. """
#     os.environ['MASTER_ADDR'] = '127.0.0.1'
#     os.environ['MASTER_PORT'] = '29500'
#     os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)  # Ensure each process only sees one GPU
#     dist.init_process_group(backend, rank=rank, world_size=size)
#     fn(rank, size)
#     dist.destroy_process_group()

# def run(rank, size):
#     step = 10 if rank != 0 else 20
#     result = verify_all_rank_step_consistent(None, step)
#     print(f"Rank {rank}: Step consistent: {result}")

# if __name__ == "__main__":
#     size = 4  # Number of GPUs
#     processes = []
#     for rank in range(size):
#         p = Process(target=init_process, args=(rank, size, run))
#         p.start()
#         processes.append(p)

#     for p in processes:
#         p.join()