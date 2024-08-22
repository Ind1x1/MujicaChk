import sys
import os

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_path)

from dataclasses import dataclass
from typing import Tuple, Callable, Any, Mapping, List

from MujicaChk.engine.shmengine import SharedMemoryEngine

import torch

shm = SharedMemoryEngine(local_rank=0)

shm.unlink()