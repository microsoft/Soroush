from collections import defaultdict
from datetime import datetime

import numpy as np

from utilities import constants
from alg import waterfilling_utils
from ncflow.lib.problem import Problem

MODEL_TIME = "model"
COMPUTATION_TIME = "computation"
EXTRACT_RATE = "extract_rate"


def get_max_min_fair_multi_path_constrained_k_waterfilling(problem: Problem, num_paths_per_flow, link_cap,
                                                           k="inf", path_characteristics=None, path_edge_idx_mapping=None,
                                                           path_path_len_mapping=None, break_down=False):
    run_time_dict = dict()
    run_time_dict[MODEL_TIME] = 0
    run_time_dict[EXTRACT_RATE] = 0
    run_time_dict[COMPUTATION_TIME] = 0

    assert k == "inf" or k == int(k)
    st_time = datetime.now()
    if path_characteristics is None:
        path_characteristics = waterfilling_utils.get_vectorized_path_characteristics(problem, path_edge_idx_mapping,
                                                                                      path_path_len_mapping,
                                                                                      num_paths_per_flow)
    sub_fid_path_len_vector, edge_idx_vector, demand_vector, path_exist_vector, num_sub_flows = path_characteristics
    output = waterfilling_utils.get_routing_matrix(problem, num_paths_per_flow, sub_fid_path_len_vector,
                                                   edge_idx_vector, demand_vector, num_sub_flows, link_cap,
                                                   return_details=False)
    routing_matrix, capacity_vector = output
    if break_down:
        checkpoint_1_time = datetime.now()
    # print(f"creating traffic;", (datetime.now() - st_time).total_seconds())
    list_commodities = problem.sparse_commodity_list
    num_flows = len(problem.sparse_commodity_list) * num_paths_per_flow
    per_flow_rate = np.zeros(shape=num_flows)
    flow_id_to_flow_rate_mapping = defaultdict(list)
    num_flows_per_link = routing_matrix @ np.ones(shape=num_flows)
    final_flow_rate = np.empty(shape=num_flows)
    np_flow_list = np.arange(num_flows)

    num_iterations = 0

    run_time_dict[MODEL_TIME] = (datetime.now() - st_time).total_seconds()
    computation_st_time = datetime.now()

    while num_flows_per_link.shape[0] and np.max(num_flows_per_link) > 0:
        num_iterations += 1
        current_fair_share = capacity_vector / num_flows_per_link
        if k == "inf":
            min_fair = np.full_like(current_fair_share, np.min(current_fair_share))
            min_link = np.where(current_fair_share <= min_fair + constants.O_epsilon)
        else:
            s_sparse = routing_matrix @ routing_matrix.transpose()
            min_link = waterfilling_utils.min_neighbor_fair_share(s_sparse, current_fair_share)

        link_mask = np.full(current_fair_share.shape, fill_value=True)
        flow_mask = np.full(routing_matrix.shape[1], fill_value=True)
        for lid, rate in min_link:
            link_mask[lid] = False
            desired_flows = routing_matrix.getrow(lid).indices
            per_flow_rate[desired_flows] = rate
            final_flow_rate[np_flow_list[desired_flows]] = rate
            flow_mask[desired_flows] = False

        capacity_vector = np.compress(link_mask, capacity_vector)
        routing_matrix = routing_matrix[link_mask]
        capacity_vector -= routing_matrix @ per_flow_rate
        routing_matrix = routing_matrix[:, flow_mask]
        np_flow_list = np.compress(flow_mask, np_flow_list)
        per_flow_rate = np.compress(flow_mask, per_flow_rate)

        num_flows_per_link = routing_matrix @ np.ones(shape=per_flow_rate.shape[0])
        while num_flows_per_link.shape[0] and np.any(num_flows_per_link == 0):
            mask = (num_flows_per_link > 0)
            routing_matrix = routing_matrix[mask]
            capacity_vector = np.compress(mask, capacity_vector)
            num_flows_per_link = routing_matrix @ np.ones(shape=per_flow_rate.shape[0])

    run_time_dict[COMPUTATION_TIME] = (datetime.now() - computation_st_time).total_seconds()

    if break_down:
        checkpoint_2_time = datetime.now()
        dur = ((checkpoint_1_time - st_time).total_seconds(), (checkpoint_2_time - checkpoint_1_time).total_seconds())
    else:
        dur = (datetime.now() - st_time).total_seconds()

    extract_st_time = datetime.now()
    for fid, (src, dst, demand) in list_commodities:
        sub_fid_list = np.arange(fid * num_paths_per_flow, (fid + 1) * num_paths_per_flow)
        flow_id_to_flow_rate_mapping[(src, dst)] = final_flow_rate[sub_fid_list]
        assert sum(flow_id_to_flow_rate_mapping[src, dst]) <= demand + constants.O_epsilon
    run_time_dict[EXTRACT_RATE] = (datetime.now() - extract_st_time).total_seconds()
    print(f"{k}-waterfilling num iterations;", num_iterations, dur)
    print(f"{k}-waterilling detailed run time {run_time_dict}")
    return flow_id_to_flow_rate_mapping, dur, run_time_dict

