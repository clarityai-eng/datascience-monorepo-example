"""Module to provide small utilites functions when interating with files"""
from pathlib import Path
from urllib.parse import urlparse


def create_dir_if_not_exists(file_path: str) -> bool:
    """Creates the parent directory if it does not exist and is a local path

    Args:
        file_path (str): The file path

    Returns:
        bool: True if the directory was created, False otherwise.
    """
    if check_is_local_path(file_path):
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        return True
    return False


def check_is_local_path(file_path: str) -> bool:
    """Checks if the file path is a local path if not it is a remote path

    Args:
        file_path (str): The file path
    Returns:
        bool: True if the file path is a local path
    """
    parsed = urlparse(file_path)
    return parsed.scheme == ""
