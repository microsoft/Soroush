import pickle
import os
import uuid
import datetime

from scipy.stats import gmean


def write_to_file(in_obj, dir_path, file_path):
    ensure_dir(dir_path)
    with open(file_path, "a") as fp:
        fp.write(in_obj)


def ensure_dir(file_path):
    try:
        os.makedirs(file_path)
    except FileExistsError:
        pass


def list_to_string(my_list, separator=","):
    str_path = str(my_list[0])
    for i in range(1, len(my_list)):
        str_path += separator + str(my_list[i])
    return str_path


def string_to_list(my_str, separator=","):
    str_list = my_str.split(separator)
    return str_list


def get_fid():
    return datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + "_" + uuid.uuid4().hex[:8]


def get_geometric_mean(data_list):
    return gmean(data_list)


def write_to_file_as_pickle(in_obj, dir_path, file_path):
    ensure_dir(dir_path)
    with open(file_path, "wb") as fp:
        pickle.dump(in_obj, fp)


def read_pickle_file(file_path):
    with open(file_path, "rb") as fp:
        output = pickle.load(fp)
    return output
