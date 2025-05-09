import logging
import torch
import torch.cuda as cuda
import warnings
import numpy as np

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("training.log", mode="a"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def setup_device():
    # Check NumPy version
    np_version = np.__version__
    if np_version.startswith("2"):
        warnings.warn("NumPy 2.x detected. This may cause issues with PyTorch. Consider downgrading to 'numpy<2'.")

    # Check for CUDA availability
    logger = setup_logging()
    if not cuda.is_available():
        logger.warning("CUDA is not available. Falling back to CPU.")
    device = torch.device("cuda" if cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    return device