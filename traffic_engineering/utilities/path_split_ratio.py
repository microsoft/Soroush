import numpy as np

from utilities import constants
from ncflow.lib import problem


def get_exponential_decay_split_ratio(path_list, base, num_paths):
    split_ratio = np.zeros(num_paths)
    num_avail_paths = len(path_list)
    coeff = np.array([np.power(base, i) for i in range(num_avail_paths)])
    split_ratio[:num_avail_paths] = coeff / np.sum(coeff)
    return split_ratio


def get_exponential_decay_len_split_ratio(path_list, base, num_paths):
    split_ratio = np.zeros(num_paths)
    num_avail_paths = len(path_list)
    coeff = np.array([np.power(base, len(path)) for path in path_list])
    split_ratio[:num_avail_paths] = coeff / np.sum(coeff)
    return split_ratio


def get_split_ratios_specific_path(path_list, base_split, split_type, num_paths):
    if split_type == constants.EXPONENTIAL_DECAY:
        return get_exponential_decay_split_ratio(path_list, base_split, num_paths)
    elif split_type == constants.EXPONENTIAL_DECAY_LEN:
        return get_exponential_decay_len_split_ratio(path_list, base_split, num_paths)
    raise Exception

