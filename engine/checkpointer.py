
from abc import ABCMeta,abstractmethod

class Checkpointer(metaclass=ABCMeta):
    """
    Base class
    """
    @abstractmethod
    def save_checkpoint(
        self, step, state_dict, path
    ):
        """
        Save the checkpoint of model, optimizer and sampler.

        Args:
            step (int): the global iteration step.
            state_dict (dict): the state dict of model and optimizer to save.
            path (str): the storage path to save the state dict.
                Note, the path is used to save the state dict to storage
                only if the training process fails.
            storage_type (StorageType): StorageType.MEMORY or StorageType.DISK.
        """
        pass

    @abstractmethod
    def load_checkpoint(self, resuming_path=None):
        """
        The manager loads the states from the files in the
        checkpoint directory to the model, optimizer and sampler.
        Args:
            resuming_path (str, optional): The manager will load checkpoint
                from the path. If the path is None, the manager will load
                the state checkpoint from the file with the maximum step.
        Return:
            A dict: a state dict.
        """
        pass