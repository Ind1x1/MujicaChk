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

from utils import env_utils
from utils.log import default_logger as logger
from engine.shmengine import (
    MUJICA_CKPT_CONFIG_KEY,
    SharedMemoryEngine,
    CheckpointConfig,
    SharedMemoryObjectPrefix
)

def compare_state_dicts(state_dict1, state_dict2):
    """
    比较两个模型的state_dict是否一致。

    参数:
        state_dict1: 第一个模型的state_dict。
        state_dict2: 第二个模型的state_dict。

    返回:
        bool: 如果两个state_dict一致，返回True；否则返回False。
    """
    # 首先比较两个state_dict的键是否一致
    if state_dict1.keys() != state_dict2.keys():
        print("State dict keys do not match.")
        return False
    
    # 比较每个键对应的张量是否相等
    for key in state_dict1.keys():
        tensor1 = state_dict1[key]
        tensor2 = state_dict2[key]
        
        if not torch.equal(tensor1, tensor2):
            print(f"Mismatch found at {key}.")
            return False

    # 如果所有键和值都匹配，返回True
    return True

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

    def __repr__(self):
        # 自定义打印格式
        return (f"{self.__class__.__name__}(\n"
                f"  fc1: {self.fc1},\n"
                f"  relu: {self.relu},\n"
                f"  dropout: {self.dropout},\n"
                f"  fc2: {self.fc2},\n"
                f"  fc3: {self.fc3},\n"
                f"  custom_tensor: {self.custom_tensor},\n"
                f"  custom_bias: {self.custom_bias}\n"
                f")")

# 示例使用
input_size = 784
hidden_size = 128
num_classes = 10

model = MLP(input_size, hidden_size, num_classes)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 打印模型实例
print(model.state_dict())

checkpoint_engine = CheckpointEngine(checkpoint_dir="./checkpoints", comm_backend="nccl")
checkpoint_engine._local_rank = env_utils.get_local_rank
checkpoint_engine._rank = 0
checkpoint_engine._group_rank = 0
checkpoint_engine.save_state_dict_to_memory(model.state_dict(), CheckpointConfig())

resotr_dict = {}
resotr_dict = checkpoint_engine.get_state_dict_from_memory()
model2 = MLP(input_size, hidden_size, num_classes)

# 将 state_dict 加载到模型中
model2.load_state_dict(resotr_dict)
model2.to(device)
print(model2.state_dict())

are_equal = compare_state_dicts(model.state_dict(), model2.state_dict())
print(f"Are the state_dicts equal? {are_equal}")