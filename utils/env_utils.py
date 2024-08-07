import os

#from MujicaChk.common.constants import NodeEnvConstant

def get_local_world_size():
    """Get the local world size."""
    return int(os.getenv("LOCAL_WORLD_SIZE",1))

def get_local_rank():
    return int(os.getenv("LOCAL_RANK", 0))

def get_rank():
    return int(os.getenv("RANK", 0))

def get_group_world_size():
    return int(os.getenv("GROUP_WORLD_SIZE", 1))

def get_group_rank():
    return int(os.getenv("GROUP_RANK", 1))

def get_env(env_key):
    """Get the specified environment variable."""
    env_value = os.getenv(env_key, None)
    return env_value