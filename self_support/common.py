import os
import inspect
try:
    import cPickle as pickle
except ImportError:
    import pickle
from collections import namedtuple

import dataset_readers


def get_package_path():
    """
    Returns the folder path that the package lies in.
    :return: folder_path: The package folder path.
    """
    return os.path.dirname(inspect.getfile(dataset_readers))


def dict_to_struct(obj):
    obj = namedtuple("Configuration", obj.keys())(*obj.values())
    return obj


def make_dirs_safe(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
