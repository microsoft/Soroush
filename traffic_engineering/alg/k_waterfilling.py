from collections import defaultdict
from datetime import datetime

import numpy as np

from utilities import constants
from alg import waterfilling_utils
from ncflow.lib.problem import Problem


def get_max_min_fair_multi_path_constrained_k_waterfilling(problem: Problem, path_edge_idx_mapping, num_paths_per_flow,
                                                           link_cap, k="inf", break_down=False):
    st_time = datetime.now()
    # assert k == "inf" or k == int(k)
    output = waterfilling_utils.get_routing_matrix(problem, path_edge_idx_mapping, num_paths_per_flow, link_cap)
    routing_matrix, capacity_vector, fid_to_sub_fid_mapping = output
    if break_down:
        checkpoint_1_time = datetime.now()
    # print(f"creating traffic;", (datetime.now() - st_time).total_seconds())
    num_flows = len(problem.sparse_commodity_list) * num_paths_per_flow
    per_flow_rate = np.zeros(shape=num_flows)
    flow_id_to_flow_rate_mapping = defaultdict(list)
    num_flows_per_link = routing_matrix @ np.ones(shape=num_flows)
    final_flow_rate = np.empty(shape=num_flows)
    np_flow_list = np.arange(num_flows)

    num_iterations = 0
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

    if break_down:
        checkpoint_2_time = datetime.now()
        dur = ((checkpoint_1_time - st_time).total_seconds(), (checkpoint_2_time - checkpoint_1_time).total_seconds())
    else:
        dur = (datetime.now() - st_time).total_seconds()
    for fid, (src, dst, demand) in problem.sparse_commodity_list:
        for sub_fid in fid_to_sub_fid_mapping[fid]:
            flow_id_to_flow_rate_mapping[(src, dst)].append(final_flow_rate[sub_fid])
        assert sum(flow_id_to_flow_rate_mapping[src, dst]) <= demand + constants.O_epsilon
    print(f"{k}-waterfilling num iterations;", num_iterations, dur)
    return flow_id_to_flow_rate_mapping, dur

