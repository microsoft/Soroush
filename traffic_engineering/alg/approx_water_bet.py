from collections import defaultdict
from datetime import datetime

import numpy as np
from scipy.sparse import csr_matrix

from alg import waterfilling_utils
from alg import approx_water_bet_old, waterfilling_utils_old
from utilities import constants
from ncflow.lib.problem import Problem


# def get_routing_matrix(list_commodities, path_edge_idx_mapping, path_path_len_mapping, path_total_path_len_mapping,
#                        row_list, column_list, num_tokens, num_rows, num_cols, fid_to_sub_fid_mapping,
#                        split_ratio_mapping, bias_toward_low_flow_rates=False, bias_flow_rate=None, bias_alpha=None):
#     data_list = np.empty_like(row_list, dtype=np.int32)
#     weight_matrix = np.empty(num_cols)
#     last_id = 0
#     if bias_toward_low_flow_rates:
#         max_flow_rate = np.average(bias_flow_rate)
#         bias_coeff = 0.000001 + np.power(bias_alpha, bias_flow_rate / max_flow_rate)
#
#     for fid, (src, dst, demand) in list_commodities:
#         const_factor = 1
#         if bias_toward_low_flow_rates:
#             const_factor *= bias_coeff[fid]
#
#         pid_to_idx_mapping = path_edge_idx_mapping[src, dst]
#         num_paths = len(pid_to_idx_mapping)
#         split_ratio = split_ratio_mapping[src, dst][:num_paths]
#         rem_tok = num_tokens - num_paths
#
#         virt_path_list = (1 + np.rint(split_ratio * rem_tok)) * const_factor
#         path_len = path_total_path_len_mapping[src, dst]
#
#         data_list[last_id:last_id + path_len] = np.repeat(virt_path_list, path_path_len_mapping[src, dst])
#         last_id += path_len
#         data_list[last_id:last_id + num_paths] = virt_path_list
#         last_id += num_paths
#
#         sub_fid_list = fid_to_sub_fid_mapping[fid]
#         weight_matrix[sub_fid_list] = virt_path_list
#
#     routing_matrix = csr_matrix((data_list, (row_list, column_list)),
#                                 shape=(num_rows, num_cols))
#     return routing_matrix, weight_matrix

def get_routing_matrix(sub_fid_path_len_vector, num_paths_per_flow, row_list, column_list, num_tokens, num_rows,
                       num_cols, split_ratio_mapping, bias_toward_low_flow_rates=False, bias_flow_rate=None, bias_alpha=None):

    bias_coeff = num_tokens
    if bias_toward_low_flow_rates:
        max_flow_rate = np.average(bias_flow_rate)
        bias_coeff *= (0.5 + np.repeat(np.power(bias_alpha, bias_flow_rate / max_flow_rate), repeats=num_paths_per_flow))

    weight_matrix = split_ratio_mapping.flatten() * bias_coeff
    data_list_1 = np.repeat(weight_matrix, repeats=sub_fid_path_len_vector)
    data_list_2 = weight_matrix
    data_list = np.concatenate((data_list_1, data_list_2))

    routing_matrix = csr_matrix((data_list, (row_list, column_list)),
                                shape=(num_rows, num_cols))
    return routing_matrix


def get_rates(problem: Problem, path_split_ratio_mapping, num_paths_per_flow, num_iter_approx_water, num_iter_bet,
              link_cap, path_characteristics=None, path_edge_idx_mapping=None, path_path_len_mapping=None,
              break_down=False, bias_toward_low_flow_rate=False, bias_alpha=None, return_matrix=False):
    assert num_iter_approx_water == 1
    st_time = datetime.now()
    if path_characteristics is None:
        path_characteristics = waterfilling_utils.get_vectorized_path_characteristics(problem, path_edge_idx_mapping,
                                                                                      path_path_len_mapping,
                                                                                      num_paths_per_flow)
    sub_fid_path_len_vector, edge_idx_vector, demand_vector, path_exist_vector, num_sub_flows = path_characteristics
    list_commodities = problem.sparse_commodity_list
    num_flows = len(list_commodities)
    num_tokens = num_paths_per_flow
    split_ratios = waterfilling_utils.get_split_ratios_all_flows(list_commodities, path_split_ratio_mapping,
                                                                 num_flows, num_paths_per_flow)
    bias_flow_rate = None
    if bias_toward_low_flow_rate:
        bias_flow_rate = np.ones(shape=len(list_commodities))
    flow_id_to_flow_rate_mapping = defaultdict(list)

    output = waterfilling_utils.get_routing_matrix(problem, num_paths_per_flow, sub_fid_path_len_vector,
                                                   edge_idx_vector, demand_vector, num_sub_flows, link_cap,
                                                   return_details=True)
    row_list, column_list, capacity_vector_cop, num_rows = output

    ones_flows = np.ones(shape=num_sub_flows)
    per_flow_rate = np.empty(num_sub_flows) * path_exist_vector
    all_satisfied = False
    
    if break_down:
        finding_split_ratios_dur = (datetime.now() - st_time).total_seconds()
        routing_matrix_dur = 0
        computation_dur = 0
        updating_split_ratios_dur = 0

    for iter_bet_no in range(num_iter_bet):
        if break_down:
            checkpoint = datetime.now()
        routing_matrix = get_routing_matrix(sub_fid_path_len_vector, num_paths_per_flow, row_list, column_list,
                                            num_tokens, num_rows, num_sub_flows, split_ratios,
                                            bias_toward_low_flow_rate, bias_flow_rate, bias_alpha=bias_alpha)

        # if num_iter_approx_water > 1:
        #     capacity_vector = capacity_vector_cop.copy()
        # else:
        capacity_vector = capacity_vector_cop

        if break_down:
            routing_matrix_dur += (datetime.now() - checkpoint).total_seconds()
            checkpoint = datetime.now()

        # per_flow_rate.fill(link_cap)
        per_flow_rate[path_exist_vector] = link_cap
        # if some_less_paths:
        #     np.multiply(per_flow_rate, path_exist_vector, out=per_flow_rate)

        num_flows_per_link = routing_matrix @ ones_flows
        # unfreezed_flows = np.ones(shape=num_sub_flows)
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

        throughput = np.reshape(final_flow_rate, (num_flows, num_paths_per_flow))
        total_rate = np.add.reduce(throughput, axis=1)

        all_satisfied = np.logical_and.reduce(total_rate >= demand_vector - constants.O_epsilon)
        if break_down:
            updating_split_ratios_dur += (datetime.now() - checkpoint).total_seconds()
        if all_satisfied:
            print("approx-bet found a satisfying solution for all demands!")
            break
            
        if bias_toward_low_flow_rate:
            bias_flow_rate = total_rate
        split_ratios = throughput / total_rate.reshape(num_flows, -1)


    if break_down:
        dur = (finding_split_ratios_dur, routing_matrix_dur, computation_dur, updating_split_ratios_dur)
    else:
        dur = (datetime.now() - st_time).total_seconds()

    if return_matrix:
        output = final_flow_rate.reshape(num_flows, num_paths_per_flow)
        return output, dur, all_satisfied

    for fid, (src, dst, demand) in list_commodities:
        sub_fid_list = np.arange(fid * num_paths_per_flow, (fid + 1) * num_paths_per_flow)
        flow_id_to_flow_rate_mapping[(src, dst)] = final_flow_rate[sub_fid_list]

    print(f"approx-waterfilling;", dur)
    return flow_id_to_flow_rate_mapping, dur, all_satisfied


def _apply_congestion(routing_matrix, flow_rate, non_zeros_fids, link_cap, update_rate):
    flow_rate_on_link = flow_rate[non_zeros_fids]
    if np.add.reduce(flow_rate_on_link) <= link_cap or flow_rate_on_link.shape[0] == 0:
        return np.inf
    mask = np.arange(flow_rate_on_link.shape[0])
    while mask.shape[0]:
        num_flows = np.add.reduce(routing_matrix[mask])
        if num_flows == 0:
            break
        fair_share = link_cap / num_flows
        allocated_rates = fair_share * routing_matrix[mask]
        under_flows = (flow_rate_on_link[mask] >= allocated_rates)
        if np.logical_and.reduce(under_flows):
            flow_rate_on_link[mask] = allocated_rates
            break
        else:
            link_cap -= flow_rate_on_link[mask] @ (1 - under_flows)
            mask = np.compress(under_flows, mask)
    if update_rate:
        flow_rate[non_zeros_fids] = flow_rate_on_link
    return fair_share
