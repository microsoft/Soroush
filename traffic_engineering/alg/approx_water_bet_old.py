from collections import defaultdict
from datetime import datetime

import numpy as np
from scipy.sparse import csr_matrix

from alg import waterfilling_utils_old as waterfilling_utils
from utilities import constants
from ncflow.lib.problem import Problem


def get_routing_matrix(list_commodities, path_edge_idx_mapping, path_path_len_mapping, path_total_path_len_mapping,
                         row_list, column_list, num_tokens, num_rows, num_cols, split_ratio_mapping):
    data_list = np.empty_like(row_list, dtype=np.float64)
    last_id = 0
    for fid, (src, dst, demand) in list_commodities:
        pid_to_idx_mapping = path_edge_idx_mapping[src, dst]
        num_paths = len(pid_to_idx_mapping)
        split_ratio = split_ratio_mapping[src, dst][:num_paths]
        virt_path_list = split_ratio * num_tokens
        path_len = path_total_path_len_mapping[src, dst]
        data_list[last_id:last_id + path_len] = np.repeat(virt_path_list, path_path_len_mapping[src, dst][:num_paths])
        last_id += path_len
        data_list[last_id:last_id + num_paths] = virt_path_list
        last_id += num_paths

    routing_matrix = csr_matrix((data_list, (row_list, column_list)),
                                shape=(num_rows, num_cols))
    return routing_matrix


def get_rates(problem: Problem, path_edge_idx_mapping, path_path_len_mapping, path_total_path_len_mapping,
              path_split_ratio_mapping, num_paths_per_flow, num_iter_approx_water, num_iter_bet, link_cap,
              break_down=False):
    assert num_iter_approx_water == 1
    st_time = datetime.now()
    list_commodities = problem.sparse_commodity_list
    num_tokens = num_paths_per_flow * 1000
    split_ratios = waterfilling_utils.get_split_ratios_all_flows(list_commodities, path_split_ratio_mapping)
    flow_id_to_flow_rate_mapping = defaultdict(list)

    output = waterfilling_utils.get_routing_matrix(problem, path_edge_idx_mapping, num_paths_per_flow, link_cap, return_details=True)
    row_list, column_list, capacity_vector_cop, fid_to_sub_fid_mapping, num_rows, num_flows, weight_vector = output
    ones_flows = np.ones(shape=num_flows)

    if break_down:
        finding_split_ratios_dur = (datetime.now() - st_time).total_seconds()
        routing_matrix_dur = 0
        computation_dur = 0
        updating_split_ratios_dur = 0

    for iter_bet_no in range(num_iter_bet):
        # print(f"iter bet {iter_bet_no}")
        if break_down:
            checkpoint = datetime.now()
        routing_matrix = get_routing_matrix(list_commodities, path_edge_idx_mapping, path_path_len_mapping,
                                            path_total_path_len_mapping, row_list, column_list, num_tokens, num_rows, num_flows, split_ratios)
        # routing_matrix = output
        # if num_iter_approx_water > 1:
        #     capacity_vector = capacity_vector_cop.copy()
        # else:
        capacity_vector = capacity_vector_cop

        if break_down:
            routing_matrix_dur += (datetime.now() - checkpoint).total_seconds()
            checkpoint = datetime.now()

        per_flow_rate = np.full(num_flows, constants.INFINITE_FLOW_RATE * 1.0) * weight_vector
        # final_flow_rate = np.empty(shape=num_flows)
        # np_flow_list = np.arange(num_flows)

        num_flows_per_link = routing_matrix @ ones_flows
        # unfreezed_flows = np.ones(shape=num_flows)
        # for _ in range(num_iter_approx_water - 1):
        #     current_fair_share = capacity_vector / num_flows_per_link
        #     s_sparse = routing_matrix @ routing_matrix.transpose()
        #     min_link = waterfilling_utils.min_neighbor_fair_share(s_sparse, current_fair_share)
        #     bottlenecked_flows = np.ones_like(per_flow_rate)
        #     link_mask = np.full(current_fair_share.shape, fill_value=True)
        #     for lid, rate in min_link:
        #         link_mask[lid] = False
        #         desired_flows = routing_matrix.getrow(lid).indices
        #         bottlenecked_flows[desired_flows] = 0
        #         per_flow_rate[desired_flows] = rate * routing_matrix[lid, desired_flows]
        #         final_flow_rate[np_flow_list[desired_flows]] = per_flow_rate[desired_flows]
        #         unfreezed_flows[np_flow_list[desired_flows]] = 0
        #
        #     capacity_vector = np.compress(link_mask, capacity_vector)
        #     routing_matrix = routing_matrix[link_mask]
        #     capacity_vector -= routing_matrix @ (per_flow_rate * (1 - bottlenecked_flows))
        #     routing_matrix = routing_matrix[:, bottlenecked_flows > 0]
        #     np_flow_list = np.compress(bottlenecked_flows, np_flow_list)
        #     per_flow_rate = np.compress(bottlenecked_flows, per_flow_rate)
        #
        #     num_flows_per_link = routing_matrix @ np.ones(shape=per_flow_rate.shape[0])
        #     while num_flows_per_link.shape[0] and np.any(num_flows_per_link == 0):
        #         mask = (num_flows_per_link > 0)
        #         routing_matrix = routing_matrix[mask]
        #         capacity_vector = np.compress(mask, capacity_vector)
        #         num_flows_per_link = routing_matrix @ np.ones(shape=per_flow_rate.shape[0])

        current_fair_share = capacity_vector / num_flows_per_link
        sorted_idx = np.argsort(current_fair_share)
        idx = 0
        num_iter = sorted_idx.shape[0]
        while idx < num_iter:
            lid = sorted_idx[idx]
            start, stop = routing_matrix.indptr[lid],  routing_matrix.indptr[lid + 1]
            non_zero_indices = routing_matrix.indices[start:stop]
            _apply_congestion(routing_matrix.data[start:stop], per_flow_rate, non_zero_indices, capacity_vector[lid], update_rate=True)
            idx += 1

        # final_flow_rate[unfreezed_flows > 0] = per_flow_rate
        final_flow_rate = per_flow_rate
        if break_down:
            computation_dur += (datetime.now() - checkpoint).total_seconds()
            checkpoint = datetime.now()

        if iter_bet_no == num_iter_bet - 1:
            break

        all_satisfied = True
        for fid, (src, dst, demand) in list_commodities:
            sub_fid_list = fid_to_sub_fid_mapping[fid]
            total_flow = np.sum(final_flow_rate[sub_fid_list])
            split_ratios[(src, dst)] = final_flow_rate[sub_fid_list] / total_flow
            if total_flow < demand - constants.O_epsilon:
                all_satisfied = False
        if break_down:
            updating_split_ratios_dur += (datetime.now() - checkpoint).total_seconds()
        if all_satisfied:
            break

    if break_down:
        dur = (finding_split_ratios_dur, routing_matrix_dur, computation_dur, updating_split_ratios_dur)
    else:
        dur = (datetime.now() - st_time).total_seconds()

    for fid, (src, dst, demand) in list_commodities:
        sub_fid_list = fid_to_sub_fid_mapping[fid]
        for pid in path_edge_idx_mapping[src, dst]:
            if len(path_edge_idx_mapping[src, dst][pid]) <= 0:
                if final_flow_rate[sub_fid_list[pid]] > 0:
                    print(src, dst, path_edge_idx_mapping[src, dst][pid], final_flow_rate[sub_fid_list[pid]])
                # assert final_flow_rate[sub_fid_list[pid]] == 0
        flow_id_to_flow_rate_mapping[(src, dst)] = final_flow_rate[sub_fid_list]

    print(f"approx-waterfilling;", dur)
    return flow_id_to_flow_rate_mapping, dur


def _apply_congestion(routing_matrix, flow_rate, non_zeros_fids, link_cap, update_rate):
    flow_rate_on_link = flow_rate[non_zeros_fids]
    if np.sum(flow_rate_on_link) <= link_cap or flow_rate_on_link.shape[0] == 0:
        return np.inf
    mask = np.arange(flow_rate_on_link.shape[0])
    while mask.shape[0]:
        num_flows = np.sum(routing_matrix[mask])
        if num_flows == 0:
            break
        fair_share = link_cap / num_flows
        under_flows = (flow_rate_on_link[mask] < fair_share * routing_matrix[mask])
        if ~np.any(under_flows):
            flow_rate_on_link[mask] = fair_share * routing_matrix[mask]
            break
        else:
            link_cap -= np.sum(flow_rate_on_link[mask] * under_flows)
            mask = np.compress(~under_flows, mask)
    if update_rate:
        flow_rate[non_zeros_fids] = flow_rate_on_link
    return fair_share
