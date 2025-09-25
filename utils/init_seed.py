import os
import random

import numpy as np
import torch
from torch.backends import cudnn


def set_seed(seed, deterministic=True):
    """
    Set random seed.
    Args:
        seed (int): random seed
        deterministic (bool, default=True): whether to set random seed
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True