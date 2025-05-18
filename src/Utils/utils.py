import logging
import os
import sys
from typing import List

import pandas as pd
from pynvml import (
    nvmlInit,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlShutdown,
)

logger = logging.getLogger(__name__)


def generate_logging_file(name: str, loc_to_write: str) -> logger:
    """
    Creates a new logging file each time the function is called.
    Ensures previous logs are cleared by overwriting the file.
    Separates INFO logs (stdout) from WARNING/ERROR logs (stderr).
    """
    try:
        log_file_path = os.path.join(loc_to_write, f"{name}.log")

        # Create a new logger
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)  # Capture all logs (INFO, WARNING, ERROR, etc.)
        logger.propagate = False  # Prevents duplicate logging in the root logger

        # Clear existing handlers to prevent duplicate logs
        if logger.hasHandlers():
            logger.handlers.clear()

        # File Handler (Logs everything: DEBUG, INFO, WARNING, ERROR, CRITICAL)
        file_handler = logging.FileHandler(log_file_path, mode="w")
        file_handler.setLevel(logging.DEBUG)  # Capture all log levels
        file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)

        # Stream Handler for stdout (INFO logs only)
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.INFO)  # Send only INFO logs to stdout
        stdout_handler.setFormatter(file_formatter)

        # Stream Handler for stderr (WARNING, ERROR, CRITICAL logs)
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(logging.WARNING)  # Send WARNING & ERROR logs to stderr
        stderr_handler.setFormatter(file_formatter)

        # Attach handlers
        logger.addHandler(file_handler)  # Log everything to a file
        logger.addHandler(stdout_handler)  # INFO logs → stdout
        logger.addHandler(stderr_handler)  # WARNING & ERROR logs → stderr

        logger.info("Logging started successfully.")  # This goes to stdout & file

        return logger

    except Exception as e:
        print(f"Failed to configure logging: {e}")
        return None


def saving_lists_of_locats_and_dicts(dir_list: List, dict_list: List, logger=None):
    """
    This function saves different dictionaries to different locations as Parquet files.
    """
    if logger is None:  # If no logger is provided, use module-level logger
        logger = logging.getLogger(__name__)

    for output_dir, data_dict in zip(dir_list, dict_list):
        os.makedirs(output_dir, exist_ok=True)
        for file_prefix, df in data_dict.items():
            file_path = os.path.join(output_dir, f"{file_prefix}.parquet")
            try:
                df.to_parquet(file_path, engine="pyarrow", compression="snappy")
                logger.info(f"Saved {file_prefix}.parquet to {output_dir}")
            except Exception as e:
                logger.error(
                    f"Failed to save {file_prefix}.parquet to {output_dir}: {e}",
                    exc_info=True,
                )


def load_parquet_files(input_dir: str) -> dict:
    """
    Loads all Parquet files from a specified directory into a dictionary of Pandas DataFrames.

    Parameters:
    - input_dir (str): Path to the directory containing Parquet files.

    Returns:
    - dict: Dictionary where keys are file names (without .parquet) and values are Pandas DataFrames.
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Directory not found: {input_dir}")

    parquet_dict = {}
    try:
        parquet_dict = {
            file_name.replace(".parquet", ""): pd.read_parquet(
                os.path.join(input_dir, file_name), engine="pyarrow"
            )
            for file_name in os.listdir(input_dir)
            if file_name.endswith(".parquet")
        }

        if not parquet_dict:
            print(f"No Parquet files found in: {input_dir}")
        else:
            print(f"Loaded {len(parquet_dict)} files from {input_dir}")

    except Exception as e:
        print(f"Error loading Parquet files from {input_dir}: {e}")

    return parquet_dict


def print_avail_gpu_memory():
    """
    Calls the pynvml library to check available GPU memory.
    """

    try:
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        mem_info = nvmlDeviceGetMemoryInfo(handle)
        print(f"Free GPU Memory: {mem_info.free / 1024 ** 2:.2f} MB")
    except Exception as e:
        print(f"Failed to retrieve GPU memory info: {e}")
    finally:
        nvmlShutdown()
