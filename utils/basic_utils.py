import os


def make_valid_path(name, is_dir=False, exist_ok=True):
    """
    This function make sure that a given path has all its parent directories created
    :param name: path name
    :param is_dir: is this path a directory and should be created also
    :param exist_ok: behaviour if the directory to be created is already existed
    :return: the same name passed to the function with its parents defined
    """
    if is_dir is True:
        parent_dir = name
    else:
        parent_dir = os.path.dirname(name)
    os.makedirs(parent_dir, exist_ok=exist_ok)
    return name
