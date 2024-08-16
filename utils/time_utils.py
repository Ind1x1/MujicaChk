import torch

from utils.log import default_logger as logger
from utils import env_utils

import time

def timer(func):
    def wrapper(*args, **kwargs):
        local_rank = env_utils.get_local_rank()
        start = time.time()
        result = func(*args, **kwargs)
        t = round(time.time() - start, 3)
        logger.info(
            f"Local rank {local_rank } execute {func.__name__} in {t}s."
        )
        return result

    return wrapper

def cuda_timer(func):
    def wrapper(*args, **kwargs):
        local_rank =  env_utils.get_local_rank()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        result = func(*args, **kwargs)
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)
        elapsed_time_s = round(elapsed_time_ms / 1000, 3)
        logger.info(
            f"Local rank {local_rank} execute {func.__name__} in {elapsed_time_s}s."
        )
        return result

    return wrapper
