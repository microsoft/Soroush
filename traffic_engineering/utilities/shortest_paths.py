import itertools
from collections import defaultdict
from multiprocessing import Manager, Process

import networkx as nx
import numpy as np

from utilities import constants, path_split_ratio


def all_pair_shortest_path_multi_processor(topology, edge_idx_dict, k, number_processors, base_split=1,
                                           split_type=constants.EXPONENTIAL_DECAY, return_dict=None):
    index = 0
    processor_index_src_dst_dict = dict()
    for (i, j) in itertools.combinations(topology, 2):
        index_processor = index % number_processors
        if index_processor not in processor_index_src_dst_dict:
            processor_index_src_dst_dict[index_processor] = list()
        processor_index_src_dst_dict[index_processor].append((i, j))
        index += 1

    if not return_dict:
        manager = Manager()
        return_dict = manager.dict()

    process_list = list()
    for i in processor_index_src_dst_dict:
        process_list.append(Process(target=k_shortest_path_list,
                                    args=(topology, edge_idx_dict, processor_index_src_dst_dict[i], k,
                                          base_split, split_type, return_dict, i)))

        print(i, " started!!!!")
        process_list[i].start()

    for i in process_list:
        i.join()
        print(i, " joined#################")

    path_proc_dict = dict()
    path_edge_idx_proc_dict = dict()
    path_path_len_proc_dict = dict()
    path_total_path_len_proc_dict = dict()
    path_split_proc_dict = dict()
    for i in processor_index_src_dst_dict:
        print(f"reading {i}")
        path_proc_dict[i] = return_dict[i][0]
        path_edge_idx_proc_dict[i] = return_dict[i][1]
        path_path_len_proc_dict[i] = return_dict[i][2]
        path_total_path_len_proc_dict[i] = return_dict[i][3]
        path_split_proc_dict[i] = return_dict[i][4]

    paths = dict()
    path_edge_idx_mapping = dict()
    path_path_len_mapping = dict()
    path_total_path_len_mapping = dict()
    path_split_ratio_mapping = dict()
    for i in path_proc_dict:
        for (src, dst) in path_proc_dict[i]:
            paths[(src, dst)] = path_proc_dict[i][(src, dst)]
            path_edge_idx_mapping[(src, dst)] = path_edge_idx_proc_dict[i][(src, dst)]
            path_path_len_mapping[(src, dst)] = path_path_len_proc_dict[i][(src, dst)]
            path_total_path_len_mapping[src, dst] = path_total_path_len_proc_dict[i][src, dst]
            path_split_ratio_mapping[src, dst] = path_split_proc_dict[i][src, dst]
    return paths, path_edge_idx_mapping, path_path_len_mapping, path_total_path_len_mapping, path_split_ratio_mapping


def get_edge_indices(path, edge_idx_dict):
    lid_list = []
    for link in zip(path, path[1:]):
        lid_list.append(edge_idx_dict[link])
    return lid_list


def k_shortest_path_list(topology, edge_idx_dict, list_src_dst, k, base_split, split_type,
                         return_dict, PID=0, add_reverse=True):
    paths = dict()
    path_edge_idx_mapping = defaultdict(dict)
    path_path_len_mapping = dict()
    path_total_path_len_mapping = defaultdict(int)
    path_to_split_ratios = dict()
    for (i, j) in list_src_dst:
        paths[i, j] = k_shortest_path(topology, i, j, k)
        path_to_split_ratios[i, j] = path_split_ratio.get_split_ratios_specific_path(paths[i, j], base_split, split_type, k)
        assert (i, j) not in path_path_len_mapping
        path_path_len_mapping[i, j] = np.zeros(shape=k, dtype=np.int16)
        for pid, path in enumerate(paths[i, j]):
            edge_indices = get_edge_indices(path, edge_idx_dict)
            path_edge_idx_mapping[i, j][pid] = np.array(edge_indices)
            path_path_len_mapping[i, j][pid] = len(edge_indices)
            path_total_path_len_mapping[i, j] += len(edge_indices)
        for pid in range(len(paths[i, j]), k):
            path_edge_idx_mapping[i, j][pid] = np.array([])
            path_path_len_mapping[i, j][pid] = 0

        if add_reverse:
            paths[j, i] = k_shortest_path(topology, j, i, k)
            path_to_split_ratios[j, i] = path_split_ratio.get_split_ratios_specific_path(paths[j, i], base_split, split_type, k)
            assert (j, i) not in path_path_len_mapping
            path_path_len_mapping[j, i] = np.zeros(shape=k, dtype=np.int16)
            for pid, path in enumerate(paths[j, i]):
                edge_indices = get_edge_indices(path, edge_idx_dict)
                path_edge_idx_mapping[j, i][pid] = np.array(edge_indices)
                path_path_len_mapping[j, i][pid] = len(edge_indices)
                path_total_path_len_mapping[j, i] += len(edge_indices)
            for pid in range(len(paths[j, i]), k):
                path_edge_idx_mapping[j, i][pid] = np.array([])
                path_path_len_mapping[j, i][pid] = 0

    return_dict[PID] = [paths, path_edge_idx_mapping, path_path_len_mapping,
                        path_total_path_len_mapping, path_to_split_ratios]


def k_shortest_path(topology, src, dest, k):
    return list(itertools.islice(nx.shortest_simple_paths(topology, src, dest), k))


def list_to_string(my_list, separator=","):
    str_path = str(my_list[0])
    for i in range(1, len(my_list)):
        str_path += separator + str(my_list[i])
    return str_path
