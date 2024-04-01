from collections import defaultdict
from datetime import datetime

import numpy as np

from alg import waterfilling_utils
from utilities import constants
from ncflow.lib.problem import Problem


def get_rates(problem: Problem, path_edge_idx_mapping, num_paths_per_flow, num_iter, link_cap, break_down=False):
    st_time = datetime.now()
    output = waterfilling_utils.get_routing_matrix(problem, path_edge_idx_mapping, num_paths_per_flow, link_cap)
    # print(f"creating traffic;", (datetime.now() - st_time).total_seconds())
    routing_matrix, capacity_vector, fid_to_sub_fid_mapping = output
    if break_down:
        checkpoint_1_time = datetime.now()
    num_flows = len(problem.sparse_commodity_list) * num_paths_per_flow
    per_flow_rate = np.full(num_flows, constants.INFINITE_FLOW_RATE * 1.0)
    final_flow_rate = np.empty(shape=num_flows)
    np_flow_list = np.arange(num_flows)
    flow_id_to_flow_rate_mapping = defaultdict(list)

    num_flows_per_link = routing_matrix @ np.ones(shape=num_flows)
    unfreezed_flows = np.ones(shape=num_flows)
    for i in range(num_iter - 1):
        current_fair_share = capacity_vector / num_flows_per_link
        s_sparse = routing_matrix @ routing_matrix.transpose()
        min_link = waterfilling_utils.min_neighbor_fair_share(s_sparse, current_fair_share)
        bottlenecked_flows = np.ones_like(per_flow_rate)
        link_mask = np.full(current_fair_share.shape, fill_value=True)
        for lid, rate in min_link:
            link_mask[lid] = False
            desired_flows = routing_matrix.getrow(lid).indices
            bottlenecked_flows[desired_flows] = 0
            per_flow_rate[desired_flows] = rate
            final_flow_rate[np_flow_list[desired_flows]] = rate
            unfreezed_flows[np_flow_list[desired_flows]] = 0

        capacity_vector = np.compress(link_mask, capacity_vector)
        routing_matrix = routing_matrix[link_mask]
        capacity_vector -= routing_matrix @ (per_flow_rate * (1 - bottlenecked_flows))
        routing_matrix = routing_matrix[:, bottlenecked_flows > 0]
        np_flow_list = np.compress(bottlenecked_flows, np_flow_list)
        per_flow_rate = np.compress(bottlenecked_flows, per_flow_rate)

        num_flows_per_link = routing_matrix @ np.ones(shape=per_flow_rate.shape[0])
        while num_flows_per_link.shape[0] and np.any(num_flows_per_link == 0):
            mask = (num_flows_per_link > 0)
            routing_matrix = routing_matrix[mask]
            capacity_vector = np.compress(mask, capacity_vector)
            num_flows_per_link = routing_matrix @ np.ones(shape=per_flow_rate.shape[0])

    current_fair_share = capacity_vector / num_flows_per_link
    sorted_idx = np.argsort(current_fair_share)
    idx = 0
    num_iter = sorted_idx.shape[0]
    while idx < num_iter:
        lid = sorted_idx[idx]
        start, stop = routing_matrix.indptr[lid],  routing_matrix.indptr[lid + 1]
        non_zero_indices = routing_matrix.indices[start:stop]
        _apply_congestion(per_flow_rate, non_zero_indices, capacity_vector[lid], update_rate=True)
        idx += 1
    final_flow_rate[unfreezed_flows > 0] = per_flow_rate
    if break_down:
        checkpoint_2_time = datetime.now()
        dur = ((checkpoint_1_time - st_time).total_seconds(), (checkpoint_2_time - checkpoint_1_time).total_seconds())
    else:
        dur = (datetime.now() - st_time).total_seconds()
    for fid, (src, dst, demand) in problem.sparse_commodity_list:
        for sub_fid in fid_to_sub_fid_mapping[fid]:
            flow_id_to_flow_rate_mapping[(src, dst)].append(final_flow_rate[sub_fid])
    print(f"approx-waterfilling;", dur)
    return flow_id_to_flow_rate_mapping, dur


def _apply_congestion(flow_rate, non_zeros_fids, link_cap, update_rate):
    flow_rate_on_link = flow_rate[non_zeros_fids]
    if np.sum(flow_rate_on_link) <= link_cap or flow_rate_on_link.shape[0] == 0:
        return np.inf
    mask = np.arange(flow_rate_on_link.shape[0])
    while mask.shape[0]:
        fair_share = link_cap / mask.shape[0]
        under_flows = (flow_rate_on_link[mask] < fair_share)
        if ~np.any(under_flows):
            flow_rate_on_link[mask] = fair_share
            break
        else:
            link_cap -= np.sum(flow_rate_on_link[mask] * under_flows)
            mask = np.compress(~under_flows, mask)
    if update_rate:
        flow_rate[non_zeros_fids] = flow_rate_on_link
    return fair_share


