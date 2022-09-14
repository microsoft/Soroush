from collections import defaultdict
from datetime import datetime

import numpy as np
from scipy.sparse import csr_matrix

from alg import waterfilling_utils
from alg import approx_water_bet_old, waterfilling_utils_old
from utilities import constants
from ncflow.lib.problem import Problem

MODEL_TIME = "model"
COMPUTATION_TIME = "computation"
EXTRACT_RATE = "extract_rate"


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
              break_down=False, bias_toward_low_flow_rate=False, bias_alpha=None, return_matrix=False, return_details=False):
    run_time_dict = dict()
    run_time_dict[MODEL_TIME] = 0
    run_time_dict[COMPUTATION_TIME] = 0
    run_time_dict[EXTRACT_RATE] = 0

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

    run_time_dict[MODEL_TIME] = (datetime.now() - st_time).total_seconds()
    
    if break_down:
        finding_split_ratios_dur = (datetime.now() - st_time).total_seconds()
        routing_matrix_dur = 0
        computation_dur = 0
        updating_split_ratios_dur = 0

    if return_details:
        sample_run_time = []
        sample_split_ratio = []
        sample_allocation = []
    computation_st_time = datetime.now()
    for iter_bet_no in range(num_iter_bet):
        if break_down:
            checkpoint = datetime.now()
        routing_matrix = get_routing_matrix(sub_fid_path_len_vector, num_paths_per_flow, row_list, column_list,
                                            num_tokens, num_rows, num_sub_flows, split_ratios,
                                            bias_toward_low_flow_rate, bias_flow_rate, bias_alpha=bias_alpha)

        capacity_vector = capacity_vector_cop

        if break_down:
            routing_matrix_dur += (datetime.now() - checkpoint).total_seconds()
            checkpoint = datetime.now()

        per_flow_rate[path_exist_vector] = link_cap * num_paths_per_flow
        per_flow_rate[split_ratios.flatten() == 0] = 0
        num_flows_per_link = routing_matrix @ ones_flows
        current_fair_share = capacity_vector / num_flows_per_link
        sorted_idx = np.argsort(current_fair_share)
        idx = 0
        num_iter = sorted_idx.shape[0]
        while idx < num_iter:
            lid = sorted_idx[idx]
            start, stop = routing_matrix.indptr[lid],  routing_matrix.indptr[lid + 1]
            non_zero_indices = routing_matrix.indices[start:stop]
            _apply_congestion(routing_matrix.data[start:stop], per_flow_rate, non_zero_indices, capacity_vector[lid],
                              update_rate=True)
            idx += 1

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

        if return_details:
            sample_split_ratio.append(split_ratios.copy())
            sample_allocation.append(throughput.copy())
        split_ratios = throughput / total_rate.reshape(num_flows, -1)

    run_time_dict[COMPUTATION_TIME] = (datetime.now() - computation_st_time).total_seconds()

    if break_down:
        dur = (finding_split_ratios_dur, routing_matrix_dur, computation_dur, updating_split_ratios_dur)
    else:
        dur = (datetime.now() - st_time).total_seconds()

    if return_matrix:
        output = final_flow_rate.reshape(num_flows, num_paths_per_flow)
        return output, dur, all_satisfied, run_time_dict
    if return_details:
        return sample_allocation, sample_split_ratio

    extract_st_time = datetime.now()
    for fid, (src, dst, demand) in list_commodities:
        sub_fid_list = np.arange(fid * num_paths_per_flow, (fid + 1) * num_paths_per_flow)
        flow_id_to_flow_rate_mapping[(src, dst)] = final_flow_rate[sub_fid_list]
    run_time_dict[EXTRACT_RATE] = (datetime.now() - extract_st_time).total_seconds()

    print(f"approx-waterfilling;", dur)
    print(f"approx-waterfilling detailed;", run_time_dict)
    return flow_id_to_flow_rate_mapping, dur, all_satisfied, run_time_dict


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
