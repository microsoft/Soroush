from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix

from utilities import constants
from ncflow.lib import Problem


def get_routing_matrix(problem: Problem, path_edge_idx_mapping, num_paths_per_flow, link_cap, return_details=False):
    # fid_to_sub_fid_mapping = defaultdict(list)
    fid_to_sub_fid_mapping = dict()
    list_commodities = problem.sparse_commodity_list
    num_flows = len(list_commodities)
    num_links = len(problem.edges_list)
    capacity_vector = np.full(shape=(num_links + num_flows), fill_value=link_cap)
    weight_vector = np.ones(shape=(num_flows * num_paths_per_flow))
    sub_fid = 0
    max_link_id = num_links
    row_list = []
    column_list = []
    for fid, (src, dst, demand) in list_commodities:
        pid_to_idx_mapping = path_edge_idx_mapping[src, dst]
        # sub_fid_list = fid_to_sub_fid_mapping[fid]
        sub_fid_list = []
        for pid, lid_list in pid_to_idx_mapping.items():
            row_list.append(lid_list)
            column_list.extend([sub_fid] * len(lid_list))
            # sub_fid_list.append(sub_fid)
            if len(lid_list) <= 0:
                weight_vector[sub_fid] = 0
            # else:
            sub_fid_list.append(sub_fid)
            sub_fid += 1
        fid_to_sub_fid_mapping[fid] = sub_fid_list
        column_list.extend([sfid for sfid in sub_fid_list])
        row_list.append(np.full(len(sub_fid_list), fill_value=max_link_id))
        capacity_vector[max_link_id] = demand
        max_link_id += 1
    row_list = np.concatenate(row_list)
    column_list = np.asarray(column_list, dtype=np.int64)
    data_list = np.ones_like(row_list)
    if return_details:
        return row_list, column_list, capacity_vector, fid_to_sub_fid_mapping, num_links + num_flows, sub_fid, weight_vector
    routing_matrix = csr_matrix((data_list, (row_list, column_list)),
                                shape=(num_links + num_flows, num_flows * num_paths_per_flow))
    return routing_matrix, capacity_vector, fid_to_sub_fid_mapping, weight_vector


def min_neighbor_fair_share(s_sparse, fair_share):
    x_len = s_sparse.shape[0]
    min_link = []
    for i in range(x_len):
        start, stop = s_sparse.indptr[i], s_sparse.indptr[i + 1]
        non_zero_indices = s_sparse.indices[start:stop]
        min_fair = np.min(fair_share[non_zero_indices])
        if fair_share[i] <= min_fair + constants.O_epsilon:
            min_link.append((i, fair_share[i]))
    return min_link


def get_split_ratios_all_flows(list_commodities, path_split_ratio_mapping):
    split_ratios = dict()
    for _, (src, dst, _) in list_commodities:
        split_ratios[src, dst] = np.copy(path_split_ratio_mapping[src, dst])
    return split_ratios