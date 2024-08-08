import logging

class RankLogger:
    def __init__(self):
        # Initialize the logger with a basic configuration
        self.logger = logging.getLogger("RankLogger")
        self.logger.setLevel(logging.INFO)

        # Create a console handler with a custom formatter
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - Rank %(rank)d - %(message)s')
        console_handler.setFormatter(formatter)
        
        # Add the handler to the logger
        if not self.logger.hasHandlers():
            self.logger.addHandler(console_handler)

    def info_0(self, message: str, local_rank: int = 0):
        """ Print info messages only if local_rank is 0. """
        if local_rank == 0:
            self.logger.info(message, extra={'rank': local_rank})

    def info(self, message: str, global_rank: int):
        """ Print messages with the given global rank. """
        self.logger.info(message, extra={'rank': global_rank})