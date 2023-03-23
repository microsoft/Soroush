from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix

from utilities import constants
from ncflow.lib import Problem

#
# def get_routing_matrix(problem: Problem, path_edge_idx_mapping, num_paths_per_flow, link_cap, return_details=False):
#     # fid_to_sub_fid_mapping = defaultdict(list)
#     fid_to_sub_fid_mapping = dict()
#     list_commodities = problem.sparse_commodity_list
#     num_flows = len(list_commodities)
#     num_links = len(problem.edges_list)
#     capacity_vector = np.full(shape=(num_links + num_flows), fill_value=link_cap)
#     sub_fid = 0
#     max_link_id = num_links
#     row_list = []
#     column_list = []
#     for fid, (src, dst, demand) in list_commodities:
#         pid_to_idx_mapping = path_edge_idx_mapping[src, dst]
#         # sub_fid_list = fid_to_sub_fid_mapping[fid]
#         sub_fid_list = np.empty(shape=len(pid_to_idx_mapping), dtype=np.int64)
#         for pid, lid_list in pid_to_idx_mapping.items():
#             row_list.append(lid_list)
#             column_list.extend([sub_fid] * len(lid_list))
#             # sub_fid_list.append(sub_fid)
#             sub_fid_list[pid] = sub_fid
#             sub_fid += 1
#         fid_to_sub_fid_mapping[fid] = sub_fid_list
#         column_list.extend([sfid for sfid in sub_fid_list])
#         row_list.append(np.full(len(sub_fid_list), fill_value=max_link_id))
#         capacity_vector[max_link_id] = demand
#         max_link_id += 1
#     row_list = np.concatenate(row_list)
#     column_list = np.asarray(column_list, dtype=np.int64)
#     if return_details:
#         return row_list, column_list, capacity_vector, fid_to_sub_fid_mapping, num_links + num_flows, sub_fid
#     data_list = np.ones_like(row_list)
#     routing_matrix = csr_matrix((data_list, (row_list, column_list)),
#                                 shape=(num_links + num_flows, num_flows * num_paths_per_flow))
#     return routing_matrix, capacity_vector, fid_to_sub_fid_mapping


def get_routing_matrix(problem: Problem, num_paths_per_flow, sub_fid_path_len_vector, edge_idx_vector,
                       demand_vector, total_num_sub_flows, link_cap, return_details=False):
    list_commodities = problem.sparse_commodity_list
    num_flows = len(list_commodities)
    num_links = len(problem.edges_list)
    max_link_id = num_links + num_flows

    row_list_1 = edge_idx_vector
    column_list_1 = np.repeat(np.arange(total_num_sub_flows), repeats=sub_fid_path_len_vector)

    row_list_2 = np.repeat(np.arange(num_links, max_link_id), repeats=num_paths_per_flow)
    column_list_2 = np.arange(total_num_sub_flows)

    row_list = np.concatenate((row_list_1, row_list_2))
    column_list = np.concatenate((column_list_1, column_list_2))

    capacity_vector = np.full(shape=(num_links + num_flows), fill_value=link_cap)
    capacity_vector[num_links:] = demand_vector

    if return_details:
        return row_list, column_list, capacity_vector, max_link_id
    data_list = np.ones_like(row_list)
    routing_matrix = csr_matrix((data_list, (row_list, column_list)),
                                shape=(max_link_id, total_num_sub_flows))
    return routing_matrix, capacity_vector


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


def get_split_ratios_all_flows(list_commodities, path_split_ratio_mapping, num_flows, num_paths):
    split_ratios = np.empty((num_flows, num_paths))
    for fid, (src, dst, _) in list_commodities:
        split_ratios[fid] = np.copy(path_split_ratio_mapping[src, dst])
    return split_ratios


def get_vectorized_path_characteristics(problem: Problem, path_edge_idx_mapping, path_path_len_mapping,
                                        num_paths_per_flow):
    flow_list = problem.sparse_commodity_list
    num_flows = len(flow_list)
    num_sub_flows = num_flows * num_paths_per_flow
    sub_fid_path_len_vector = np.empty(num_sub_flows, dtype=np.int32)
    demand_vector = np.empty(num_flows)
    edge_idx_vector = []
    sub_fid = 0
    for fid, (src, dst, demand) in flow_list:
        pid_edge_idx_mapping = path_edge_idx_mapping[src, dst]
        pid_path_len_mapping = path_path_len_mapping[src, dst]
        demand_vector[fid] = demand
        for pid in range(num_paths_per_flow):
            path_len = pid_path_len_mapping[pid]
            sub_fid_path_len_vector[sub_fid] = path_len
            edge_idx_vector.append(pid_edge_idx_mapping[pid])
            sub_fid += 1
    edge_idx_vector = np.concatenate(edge_idx_vector)
    path_exist_vector = (sub_fid_path_len_vector > 0)
    return sub_fid_path_len_vector, edge_idx_vector, demand_vector, path_exist_vector, sub_fid


def get_approx_label(approach, num_iter):
    if approach == constants.APPROX:
        # return approach.format(num_iter[0])
        return "approx"
    if approach == constants.APPROX_BET:
        # return approach.format(num_iter[0], num_iter[1])
        return "heuristic"
    if approach == constants.APPROX_MCF:
        # return approach.format(num_iter[0])
        return "approx_bet_mcf"
    if approach == constants.APPROX_BET_MCF:
        # return approach.format(num_iter[0], num_iter[1])
        return "Equi-depth binner"
    if approach == constants.APPROX_BET_MCF_BIASED:
        # return approach.format(num_iter[0], num_iter[1])
        return "Equi-depth binner biased"
    return approach
