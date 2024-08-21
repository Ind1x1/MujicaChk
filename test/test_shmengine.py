import sys
import os

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_path)

from dataclasses import dataclass
from typing import Tuple, Callable, Any, Mapping, List

from engine.shmengine import SharedMemoryEngine

import torch

# 示例 state_dict 包含张量和非张量
state_dict = {
    'layer1': torch.randn(3, 3),  # PyTorch张量
    'state': 'training',  # 字符串
    'optimizer': {'lr': 0.001, 'momentum': 0.9},  # 字典
    'data': [torch.randn(2, 2), torch.randn(2,3)]  # 列表，包含张量和整数
}

print("-------------------------state_dict-------------------------")
for key, value in state_dict.items():
    print(f"{key}: {value}")

shm = SharedMemoryEngine(local_rank=0)

shm.save_state_dict(state_dict=state_dict)

print("--------------------------meta_dict-------------------------")
for key, value in shm.meta_dict.items():
    print(f"{key}: {value}")

restored_state_dict = shm.load_state_dict()

print("-----------------------restored_dict------------------------")
for key, value in restored_state_dict.items():
    print(f"{key}: {value}")

