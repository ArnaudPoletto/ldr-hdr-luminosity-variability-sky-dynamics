import torch
import random
import numpy as np


def set_seed(seed: int):
    """
    Set the random seed for all relevant packages.

    Args:
        seed (int): The seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)