import os
from pathlib import Path


def load_data(path):
    """
    Grabs paths of all images
    """

    # Subdirectories for cancerous and non cancerous
    YES = "yes"
    NO = "no"

    yes_path = Path(path, YES)
    no_path = Path(path, NO)

    yes_files = list()
    no_files = list()

    for (dirpath, _, filenames) in os.walk(yes_path):
        yes_files += [os.path.join(dirpath, file) for file in filenames]
    for (dirpath, _, filenames) in os.walk(no_path):
        no_files += [os.path.join(dirpath, file) for file in filenames]

    return (yes_files, no_files)
