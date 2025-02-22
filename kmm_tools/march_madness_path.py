"""
A simple utility that allows you to override the competition data path
using an environment variable so that they same code will run whether
on kaggle or locally
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ModuleNotFoundError:
    pass
if (p := os.getenv("KAGGLE_KERNEL_RUN_TYPE")) is not None:
    DEFAULT_COMPETITION_DATA_PATH = Path(
        "/kaggle/input/march-machine-learning-mania-2025"
    )
    # COMPETITION_DATA_PATH = Path("/kaggle/input/march-machine-learning-mania-2025")
elif (p := os.getenv("COMPETITION_DATA_PATH")) is not None:
    DEFAULT_COMPETITION_DATA_PATH = Path(p)
else:
    raise RuntimeError(
        "If running locally you must define an environment variable COMPETITION_DATA_PATH."
    )
