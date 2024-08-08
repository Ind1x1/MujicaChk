import threading
from typing import Optional

def singleton(cls):
    _instance = {}
    _instance_lock = threading.Lock()

    def _singleton(*args, **kwargs):
        with _instance_lock:
            if cls not in _instance:
                _instance[cls] = cls(*args, **kwargs)
        return _instance[cls]

    return _singleton

class Singleton(object):
    _instance_lock: Optional[threading.Lock] = None
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def singleton_instance(cls, *args, **kwargs):

        if not cls._instance_lock:
            with cls._lock:
                if not cls._instance_lock:
                    cls._instance_lock = threading.Lock()

        if not cls._instance:
            with cls._instance_lock:
                if not cls._instance:
                    cls._instance = cls(*args, **kwargs)
        return cls._instance