import os
import torch
import torch.distributed as dist

def setup(rank, world_size):
    # 设置环境变量
    os.environ['MASTER_ADDR'] = 'localhost'  # 或主节点的IP地址
    os.environ['MASTER_PORT'] = '12355'  # 选择一个开放的端口号

    # 初始化进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def example_all_reduce(rank, world_size):
    setup(rank, world_size)

    # 创建一个张量并将其移动到对应的GPU
    tensor = torch.ones(10).to(rank)
    print(f"Rank {rank} starting with tensor: {tensor}")

    # 执行All-Reduce操作
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    # tensor /= world_size

    print(f"Rank {rank} has tensor after All-Reduce: {tensor}")

    cleanup()

if __name__ == "__main__":
    world_size = 4  # GPU数量
    torch.multiprocessing.spawn(example_all_reduce, args=(world_size,), nprocs=world_size, join=True)
