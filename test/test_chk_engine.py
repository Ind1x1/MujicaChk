import sys
import os

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_path)

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp

from datetime import timedelta
from typing import Optional, List

from engine.chk_engine import CheckpointEngine

from engine.shmengine import (
    MUJICA_CKPT_CONFIG_KEY,
    SharedMemoryEngine,
    CheckpointConfig,
    SharedMemoryObjectPrefix
)
from utils import env_utils
from utils.log import default_logger as logger

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

        # 添加自定义张量
        self.custom_tensor = torch.randn(hidden_size, hidden_size)
        self.custom_bias = torch.randn(hidden_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

    def custom_state_dict(self):
        state = self.state_dict()
        # 将自定义张量添加到state_dict中
        state['custom_tensor'] = self.custom_tensor
        state['custom_bias'] = self.custom_bias
        return state

    def load_custom_state_dict(self, state_dict):
        self.load_state_dict(state_dict)
        # 从state_dict中加载自定义张量
        self.custom_tensor = state_dict['custom_tensor']
        self.custom_bias = state_dict['custom_bias']
    
class SimpleCheckpointEngine(CheckpointEngine):
    def __init__(self, checkpoint_dir: str, comm_backend: str="nccl"):
        super().__init__(checkpoint_dir, comm_backend=comm_backend)
        self.state_dict ={}
        self.paths = {}

    def save(self, state_dict, path: str):
        self.save_state_dict_to_memory(state_dict, CheckpointConfig())
        logger.info(f"Checkpoint saved to shared memory for path: {path}")

    def load(self, path: str, map_location=None):
        # 简单模拟从共享内存加载的过程
        logger.info(f"Checkpoint saved to shared memory for path: {path}")
        return self._shm_handler.memory  # 模拟的加载
    
def generate_dummy_data(num_samples, input_size, num_classes):
    # 生成随机输入数据和标签
    inputs = torch.randn(num_samples, input_size)
    labels = torch.randint(0, num_classes, (num_samples,))
    return inputs, labels


def train():
    # 获取rank和world_size
    rank = int(os.getenv('RANK', -1))
    world_size = int(os.getenv('WORLD_SIZE', -1))
    local_rank = env_utils.get_local_rank()
    
    if rank == -1 or world_size == -1 or local_rank == -1:
        raise ValueError("RANK, WORLD_SIZE, or LOCAL_RANK not set properly.")
    
    try:
        logger.info(f"Process {rank} (local rank {local_rank}) started.")
        
        # 初始化进程组
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)

        # 设置模型参数
        input_size = 100
        hidden_size = 128
        num_classes = 10
        num_samples = 1000
        
        # 创建模型并将其移动到当前GPU
        model = MLP(input_size, hidden_size, num_classes).cuda(local_rank)
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        logger.info(f"{rank}: -------------> Model on GPU {local_rank}")
        print(model)
        logger.info("------------------->")

        # 创建检查点引擎
        checkpoint_engine = SimpleCheckpointEngine(checkpoint_dir="./checkpoints", comm_backend="nccl")
        checkpoint_engine._local_rank = local_rank

        # 生成训练数据
        inputs, labels = generate_dummy_data(num_samples, input_size, num_classes)
        inputs = inputs.cuda(local_rank)
        labels = labels.cuda(local_rank)

        # 模拟训练过程
        for epoch in range(5):
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            logger.info(f"Epoch [{epoch+1}/5], Loss: {loss.item():.4f}")

            # 保存模型检查点到共享内存（包括自定义张量）
            checkpoint_engine.save(model.custom_state_dict(), f"model_epoch_{epoch}_rank_{rank}.ckpt")

        # 关闭检查点引擎
        checkpoint_engine.close()

        logger.info(f"Process {rank} finished.")
        
        # 销毁进程组
        dist.destroy_process_group()

    except Exception as e:
        logger.error(f"Error in process {rank}: {e}")
        raise e

if __name__ == "__main__":
    train()