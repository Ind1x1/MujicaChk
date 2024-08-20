import torch
import torch.nn as nn
import torch.optim as optim
import deepspeed
import os
import sys
from deepspeed.runtime.zero.parameter_offload import DeepSpeedZeRoOffload
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_path)

def main():
    model_state_path = "./outputtest/global_step1/mp_rank_00_model_states.pt"
    optim_state_paths = [
        "./outputtest/global_step1/zero_pp_rank_0_mp_rank_00_optim_states.pt",
        "./outputtest/global_step1/zero_pp_rank_1_mp_rank_00_optim_states.pt",
        "./outputtest/global_step1/zero_pp_rank_2_mp_rank_00_optim_states.pt",
        "./outputtest/global_step1/zero_pp_rank_3_mp_rank_00_optim_states.pt"
    ]
    
    # 加载模型状态
    try:
        model_state = torch.load(model_state_path)
        print("Model state loaded successfully.")
        print(model_state)
    except Exception as e:
        print(f"Failed to load model state: {e}")
    
    # 加载每个优化器状态
    for idx, path in enumerate(optim_state_paths):
        try:
            optim_state = torch.load(path)
            print(f"Optimizer state {idx} loaded successfully.")
            print(optim_state)
        except Exception as e:
            print(f"Failed to load optimizer state {idx}: {e}")

if __name__ == "__main__":
    main()
