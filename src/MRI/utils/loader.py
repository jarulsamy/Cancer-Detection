# -*- coding: utf-8 -*-
import os
from pathlib import Path


def load_data(path, YES="yes", NO="no"):
    """
    Grabs paths of all images

    Arguments:
    path - Path where yes/no subdirs exist.
    YES - subdir name where YES samples exist.
    NO - subdir name where NO samples exist.
    """

    # Subdirectories for cancerous and non cancerous
    yes_path = Path(path, YES)
    no_path = Path(path, NO)

    if not os.path.isdir(yes_path):
        raise FileNotFoundError(f"Directory {yes_path} does not exist!")
    if not os.path.isdir(no_path):
        raise FileNotFoundError(f"Directory {no_path} does not exist!")

    yes_files = []
    no_files = []

    for (dirpath, _, filenames) in os.walk(yes_path):
        yes_files += [os.path.join(dirpath, file) for file in filenames]
    for (dirpath, _, filenames) in os.walk(no_path):
        no_files += [os.path.join(dirpath, file) for file in filenames]

    return (yes_files, no_files)
