import torch
import torch.nn as nn
import torch.optim as optim
import deepspeed
import os
import sys
import torch.distributed as dist
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_path)
from MujicaChk.engine.dspeed import DeepSpeedCheckpointer
from MujicaChk.utils import env_utils

# 定义一个简单的模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc(x)

def setup_distributed():
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed()

def main():
    # 初始化分布式环境
    setup_distributed()

    # 配置DeepSpeed
    ds_config = {
        "train_batch_size": 8,
        "gradient_accumulation_steps": 1,
        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "overlap_comm": True,
            "contiguous_gradients": True
        },
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-3
            }
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": 1e-3,
                "warmup_num_steps": 100
            }
        },
    }

    # 初始化模型、优化器和DeepSpeed引擎
    model = SimpleModel().cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=None,
        model=model,
        optimizer=optimizer,
        config=ds_config
    )

    # 加载检查点的目录
    load_dir = './outputtest'
    tag = None  # 加载最新的检查点
    
    MujicaCheckpointer = DeepSpeedCheckpointer(model_engine, "./outputtest") 
    MujicaCheckpointer.load_checkpoint('./outputtest')


    # try:
    #     load_path, client_state = model_engine.load_checkpoint(
    #         load_dir=load_dir,
    #         tag=tag
    #     )
    # except Exception as e:
    #     print(f"Error loading checkpoint: {e}")
    #     load_path, client_state = None, None

    # # 检查是否成功加载
    # if load_path is None:
    #     print("Checkpoint loading failed.")
    # else:
    #     print(f"Checkpoint loaded from {load_path}.")

    #     # 打印模型的状态
    #     print("Model State Dict:")
    #     for param_tensor in model_engine.module.state_dict():
    #         print(f"Rnak {dist.get_rank()} {param_tensor}\t{model_engine.module.state_dict()[param_tensor].size()}")

    #     # 打印 client_state 的内容
    #     print("\nClient State:")
    #     for key, value in client_state.items():
    #         print(f"Rnak {dist.get_rank()} {key}: {value}")
        
    #     # 打印优化器的状态
    #     print("\nOptimizer State Dict on Rank", dist.get_rank(), ":")
    #     for var_name in optimizer.state_dict():
    #         print(f"Rnak {dist.get_rank()} {var_name}\t{optimizer.state_dict()[var_name]}")

if __name__ == "__main__":
    main()
