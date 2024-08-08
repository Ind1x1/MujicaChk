import sys
import os

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_path)

from utils.rank_logger import RankLogger
# from MujicaChk.utils.rank_logger import RankLogger

def main():
    rank_logger = RankLogger()

    # Example for rank 0
    rank_logger.print_rank0_info("This is a message for rank 0.", local_rank=0)
    
    # Example for a specific rank
    rank_logger.print_info("This is a message for global rank 2.", global_rank=2)

if __name__ == "__main__":
    main()