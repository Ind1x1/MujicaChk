import torch
import torch.nn as nn
import torch.optim as optim
import deepspeed
import os
import torch.distributed as dist

# 定义一个简单的模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc(x)

def setup_distributed():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    
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
        }
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
    inputs = torch.randn(8, 10, device='cuda', dtype=torch.half)
    labels = torch.randn(8, 10, device='cuda', dtype=torch.half)

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

    # 打印state_dict的一部分以验证同步
    if torch.distributed.get_rank() == 0:
        print("State Dict of the first layer's weights:", state_dict['fc.weight'])

    # 在GPU 0 上收集其他 GPU 中的fc.weight并拼接
    fc_weight = state_dict['fc.weight'].clone()

    cmmgroup = torch.distributed.new_group(ranks=[0, 1, 2])

    # 准备一个列表来存放 GPU 1 和 GPU 2 的权重（GPU 0 不需要收集自己的权重）
    gathered_weights = None
    if torch.distributed.get_rank() == 0:
        gathered_weights = [torch.zeros_like(fc_weight) for _ in range(2)]  # 只收集 GPU 1 和 GPU 2 的权重

    # 执行 gather 操作，将 GPU 1 和 GPU 2 的权重收集到 GPU 0 上
    if torch.distributed.get_rank() in [1, 2]:
        torch.distributed.gather(fc_weight, gather_list=None, dst=0, group=cmmgroup)
    elif torch.distributed.get_rank() == 0:
        torch.distributed.gather(fc_weight, gather_list=gathered_weights, dst=0, group=cmmgroup)
    
    # 在 GPU 0 上拼接权重并输出
    if torch.distributed.get_rank() == 0:
        print("Weights collected from GPU 1:", gathered_weights[0])
        print("Weights collected from GPU 2:", gathered_weights[1])

if __name__ == "__main__":
    main()

