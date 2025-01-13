import random
import os
import numpy as np

import torch

def set_seed(seed: int) -> None:
    """
    Set the seed for random number generators in NumPy, TensorFlow, and PyTorch.

    Parameters:
    seed (int): The seed value to set for random number generators.

    Returns:
    None

    Example:
    >>> set_seed(42)
    """
    # Set seed for Python random module
    random.seed(seed)

    # Set PYTHONHASHSEED environment variable
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Set seed for NumPy
    np.random.seed(seed)

    # Set seed for PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # Disable CuDNN benchmarking for reproducibility

        # Set CUDA environment variables
        os.environ['CUDA_SEED'] = str(seed)  # Set CUDA seed



def format_time(milliseconds):
    seconds = milliseconds // 1000
    minutes = seconds // 60
    hours = minutes // 60
    return f"{hours:02}:{minutes % 60:02}:{seconds % 60:02}"


