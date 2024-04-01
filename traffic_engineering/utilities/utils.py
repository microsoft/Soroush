import pickle
import os
import uuid
import datetime

from glob import iglob
from scipy.stats import gmean


def find_topo_tm_fname(topo_name, tm_model):
    fname_list = []
    topo_fname = os.path.join('..', 'ncflow', 'topologies', 'topology-zoo', topo_name)
    for tm_fname in iglob(
            '../ncflow/traffic-matrices/{}/{}*_traffic-matrix.pkl'.format(
                tm_model, topo_name)):
        vals = os.path.basename(tm_fname)[:-4].split('_')
        _, traffic_seed, scale_factor = vals[1], int(vals[2]), float(vals[3])
        fname_list.append((topo_name, tm_model, scale_factor, topo_fname, tm_fname))
    return fname_list


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


def file_exists(file_path):
    return os.path.exists(file_path)


def revise_list_commodities(problem):
    found_zero = False
    list_to_remove = []
    for fid, (i, j, demand) in problem.sparse_commodity_list:
        if demand <= 0:
            found_zero = True
            print(f"flow {fid} with src {i} dst {j} demand {demand} removed!!")
            list_to_remove.append((fid, (i, j, demand)))

    if found_zero:
        for fid, (src, dst, demand) in list_to_remove:
            problem.sparse_commodity_list.remove((fid, (src, dst, demand)))
        for idx, (fid, (src, dst, demand)) in enumerate(problem.sparse_commodity_list):
            problem.sparse_commodity_list[idx] = (idx, (src, dst, demand))
    for fid, (i, j, demand) in problem.sparse_commodity_list:
        assert demand > 0