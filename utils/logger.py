import logging
import os
import sys

class RankFilter(logging.Filter):
    """Filter out logs from non-zero ranks."""
    def __init__(self, rank: int):
        super().__init__()
        self.rank = rank

    def filter(self, record: logging.LogRecord) -> bool:
        return self.rank == 0


def create_logger(log_file=None, rank=0):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.propagate = False  # avoid replicated print out

    # Clean handler (to avoid DDP add handler repeatedly)
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addFilter(RankFilter(rank))

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # file handler
    if log_file is not None and rank == 0:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def write_log(content, logger, rank):
    if rank == 0:
        logger.info(content)