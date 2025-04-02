import pickle
import os


def save_object(obj, file_path: str):
    """
    Saves the given object to disk using pickle.

    Parameters:
    - obj: The Python object to save.
    - file_path: The destination file path.
    """
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


def load_object(file_path: str):
    """
    Loads an object from disk using pickle.

    Parameters:
    - file_path: The source file path.

    Returns:
    - The loaded Python object, or None if the file does not exist.
    """
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)
    return None
