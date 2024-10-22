# pylint: disable=invalid-name
"""File handling, image manipulation and memory query functions."""

import subprocess
import json
from pathlib import Path
import pandas as pd
import torch
import pynvml

class Constants:
    """Project constants"""
    DATASET_FILEPATH = "./data/external"
    WB_PROJECT = "cyclegan"
    WB_DB_UPLOAD_JOB = "dataset_upload"
    WB_DB_ARTIFACT_TYPE = "datasets"


def remove_all_files(folder_path):
    """Remove all files in a folder."""
    folder = Path(folder_path)
    if folder.exists() and folder.is_dir():
        for file in folder.iterdir():
            if file.is_file():
                file.unlink()
            elif file.is_dir():
                remove_all_files(file)


def filter_dataframe(df, filter_dict):
    """Filter a DataFrame by multiple columns.

    Attributes:
    ------------
    df: pd.DataFrame
        DataFrame to filter.
    filter_dict: dict
        Dictionary with column names as keys and values to filter as values.
    """
    mask = pd.Series([True] * len(df))
    for col, values in filter_dict.items():
        mask &= df[col].isin(values)
    return df[mask]


def get_gpu_memory_usage_dict():
    """Get list of dict with memory usage of all GPUs."""
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()

    gpu_memory_info = []
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

        gpu_memory_info.append({
            'gpu_index': i,
            'total_memory': memory_info.total,
            'used_memory': memory_info.used,
            'free_memory': memory_info.free
        })
    pynvml.nvmlShutdown()
    return gpu_memory_info


def get_gpu_memory_usage(msg=None, short_msg=False):
    """Print the memory usage of all GPUs.

    Attibutes:
    ------------
    msg: str, optional
        Message to print before the memory usage. If None
        provided, the default message is "GPU Memory Usage".
        (default=None)
    short_msg: bool
        If True, prints a single line message with the total
        memory usage across all GPUs. If msg=='' and
        short_msg==True, returns the percentage of memory used.
        (default=False)
    """
    gpu_memory_info = get_gpu_memory_usage_dict()
    if short_msg:
        if msg is None:
            msg = "GPU Memory Usage"
        total = sum(info['total_memory'] for info in gpu_memory_info)
        used = sum(info['used_memory'] for info in gpu_memory_info)
        if len(msg):
            return f"{msg}: {used / (1024 ** 2):.2f} MB ({used / total * 100:.2f}% used)"
        else:
            return f'{used / total * 100:.2f}%'

    out = ""
    ident = ""
    if msg is not None:
        out = f'{msg}\n'
        ident = " " * 2
    if len(gpu_memory_info) == 0:
        out += f"{ident}No GPUs found."
    for info in gpu_memory_info:
        out += f"{ident}GPU {info['gpu_index']}:\n"
        out += f"{ident}  Total Memory: {info['total_memory'] / (1024 ** 2):.2f} MB\n"
        out += f"{ident}  Used Memory: {info['used_memory'] / (1024 ** 2):.2f} MB\n"
        out += f"{ident}  Free Memory: {info['free_memory'] / (1024 ** 2):.2f} MB"
    return out


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of parameters in a PyTorch model.

    Attributes:
    ------------
    model: nn.Module
        The PyTorch model.

    Returns:
    ------------
        The total number of parameters: int
    """
    return sum(p.numel() for p in model.parameters())


def get_current_commit():
    """Get current git hash of the repository."""
    try:
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
        commit_message = subprocess.check_output(['git', 'log', '-1', '--pretty=%B'])
        return git_hash.strip().decode('utf-8'), commit_message.strip().decode('utf-8')
    except subprocess.CalledProcessError:
        return None, None


def save_dict_as_json(data, file_path):
    """
    Save a dictionary as a formatted JSON file.

    Parameters:
    ------------
    data: dict
        The dictionary to save.
    file_path: str
        The path to the output JSON file.
    """
    out = {}
    for k,v in data.items():
        out[k] = str(v)

    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(out, json_file, indent=4, sort_keys=True)

def load_json_to_dict(file_path):
    """
    Load a JSON file into a dictionary.

    Tries to convert strings to None, bool, int or float.

    Parameters:
    ------------
    file_path: str
        The path to the JSON file.

    Returns:
    ------------
    dict
        The dictionary containing the JSON data.
    """
    with open(file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    for k,v in data.items():
        if v == 'None':
            data[k] = None
        elif v == 'True':
            data[k] = True
        elif v == 'False':
            data[k] = False
        else:
            try:
                data[k] = int(v)
            except ValueError:
                try:
                    data[k] = float(v)
                except ValueError:
                    pass
    return data
