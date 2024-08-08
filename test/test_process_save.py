import sys
import os

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_path)

from dataclasses import dataclass
from typing import Tuple, Callable, Any, Mapping, List

from multiprocessing import Process
import logging
from utils import env_utils  # 假设这个模块提供了 get_local_rank 函数

import time

# 设置 logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def start_async_save():
    # 这里是保存检查点的逻辑
    while True:
        # 假设有一些保存检查点的代码
        time.sleep(10)  # 每10秒保存一次
        logger.info("Checkpoint saved.")

def start_saver_process():
    """
    Start a process to asynchronously save checkpoint if the training
    process is not launched by `dlrover-run`. This process will
    exit and cannot save the checkpoint after the training process exit.
    It is better to use `dlrover-run` to start the training process.
    `dlrover-run` can save checkpoint once the training process fails
    and relaunch new training processes which can restore the checkpoint
    from the memory not the storage.
    """
    local_rank = env_utils.get_local_rank()
    role_name = os.getenv("ROLE_NAME", "")
    # Only start the process on local rank 0
    # if the training process is not launched by dlrover-run.
    if role_name != "dlrover-trainer" and local_rank == 0:
        p = Process(target=start_async_save, daemon=True)
        p.start()
        logger.info("Start a process to asynchronously save checkpoint.")
        return p
    return None

if __name__ == "__main__":
    # 假设这是你的训练脚本
    saver_process = start_saver_process()

    # 模拟训练过程
    for epoch in range(5):
        time.sleep(5)  # 模拟训练过程
        print(f"Epoch {epoch+1} completed.")

    # 确保异步保存进程在训练结束时正确终止
    if saver_process:
        saver_process.terminate()  # 发送终止信号
        saver_process.join()       # 等待进程完全退出
        logger.info("Checkpoint saver process terminated.")
