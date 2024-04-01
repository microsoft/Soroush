import numpy as np

from utilities import constants


def get_exponential_decay_split_ratio(num_gpus, base):
    coeff = np.array([np.power(base, i) for i in range(num_gpus)])
    split_ratio = coeff / np.sum(coeff)
    return split_ratio


def get_exponential_decay_len_split_ratio(throughput_list, base):
    coeff = np.power(base, throughput_list)
    split_ratio = coeff / np.sum(coeff)
    return split_ratio


def get_split_ratios_specific_job(throughput_list, base_split, split_type):
    if split_type == constants.EXPONENTIAL_DECAY:
        return get_exponential_decay_split_ratio(len(throughput_list), base_split)
    elif split_type == constants.EXPONENTIAL_DECAY_LEN:
        return get_exponential_decay_len_split_ratio(throughput_list, base_split)
    raise Exception
