import torch
import torch.nn as nn
import torch.optim as optim
import deepspeed
import os
import sys
from deepspeed.runtime.zero.parameter_offload import DeepSpeedZeRoOffload
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_path)

from engine.dspeed import DeepSpeedCheckpointer

# 定义一个简单的模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc(x)

def setup_distributed():
    local_rank = int(os.environ['LOCAL_RANK'])
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

    # 创建一些示例数据，并转换为FP16精度
    inputs = torch.randn(8, 10, device='cuda', dtype=torch.half)  # 推荐的方式
    labels = torch.randn(8, 10, device='cuda', dtype=torch.half)  # 推荐的方式

    # 执行一个训练步骤
    outputs = model_engine(inputs)
    loss = nn.MSELoss()(outputs, labels)
    
    # 确保损失也转换为FP16精度
    loss = loss.half()

    # 执行反向传播
    model_engine.backward(loss)
    model_engine.step()

    # 同步并获取state_dict
    state_dict = model_engine.module.state_dict()

    MujicaCheckpointer = DeepSpeedCheckpointer(model_engine, "./outputtest") 
    MujicaCheckpointer.save_checkpoint("./outputtest",1)


    # if torch.distributed.get_rank() == 0:
    #     checkpointlist = model_engine._get_all_ckpt_names("./outputtest","global_step1")
    #     print(checkpointlist)

if __name__ == "__main__":
    main()
