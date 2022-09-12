from collections import defaultdict
from datetime import datetime

import numpy as np
from scipy.sparse import diags

from alg import waterfilling_utils
from utilities import constants
from ncflow.lib.problem import Problem


MODEL_TIME = "model"
COMPUTATION_TIME = "computation"
EXTRACT_RATE = "extract_rate"


def get_rates(problem: Problem, num_paths_per_flow, link_cap, path_characteristics=None,
              path_edge_idx_mapping=None, path_path_len_mapping=None):
    run_time_dict = dict()
    run_time_dict[MODEL_TIME] = 0
    run_time_dict[COMPUTATION_TIME] = 0
    run_time_dict[EXTRACT_RATE] = 0

    st_time = datetime.now()
    if path_characteristics is None:
        path_characteristics = waterfilling_utils.get_vectorized_path_characteristics(problem, path_edge_idx_mapping,
                                                                                      path_path_len_mapping,
                                                                                      num_paths_per_flow)
    sub_fid_path_len_vector, edge_idx_vector, demand_vector, path_exist_vector, num_sub_flows = path_characteristics
    list_commodities = problem.sparse_commodity_list
    num_flows = len(list_commodities)

    output = waterfilling_utils.get_routing_matrix(problem, num_paths_per_flow, sub_fid_path_len_vector,
                                                   edge_idx_vector, demand_vector, num_sub_flows, link_cap,
                                                   return_details=False)
    routing_matrix, capacity_vector = output

    final_flow_rate = np.zeros(num_sub_flows)
    which_sub_flow_active = np.zeros(shape=(num_flows, num_paths_per_flow))
    which_sub_flow_active[:, 0] = 1
    flow_active_idx = np.zeros(shape=num_flows, dtype=np.int32)
    np_sub_flow_list = np.arange(num_sub_flows)
    already_seen_subflows = set()
    true_links = np.arange(len(problem.G.edges))
    fid_to_subflow_removed = np.zeros(shape=num_flows)
    run_time_dict[MODEL_TIME] = (datetime.now() - st_time).total_seconds()

    computation_st_time = datetime.now()
    iter_no = 0
    while np.any(which_sub_flow_active):
        iter_no += 1
        print(f"============ {routing_matrix.shape}")
        reshaped_active_sub_flow = which_sub_flow_active.reshape(-1, )[np_sub_flow_list]
        num_flows_per_link = routing_matrix @ reshaped_active_sub_flow
        rem_fair_share = capacity_vector / num_flows_per_link
        link_min_fair = np.min(rem_fair_share[true_links])

        if np_sub_flow_list.shape[0] and link_min_fair > link_cap * 10:
            break

        all_demand_constraint_flows = (rem_fair_share < link_min_fair - constants.O_epsilon)
        link_mask = np.full(capacity_vector.shape, fill_value=True)
        flow_mask = np.full(routing_matrix.shape[1], fill_value=True)
        starting_v_lid = true_links.shape[0]
        if np.any(all_demand_constraint_flows[starting_v_lid:]):
            desired_lids = starting_v_lid + np.where(all_demand_constraint_flows[starting_v_lid:])[0]
            rate = np.zeros_like(np_sub_flow_list, dtype=np.float)
            # print(np.count_nonzero(reshaped_active_sub_flow), np.count_nonzero(which_sub_flow_active), num_flows)
            # print(true_links.shape, starting_v_lid, len(problem.G.edges))
            # print(desired_lids.shape)
            rate[np.where(reshaped_active_sub_flow)] = np.minimum(np.max(capacity_vector[desired_lids]), capacity_vector[starting_v_lid:])
            final_flow_rate[np_sub_flow_list] += rate
            capacity_vector -= routing_matrix @ rate
            for lid in desired_lids:
                assert lid >= starting_v_lid
                link_mask[lid] = False
                desired_flows = np.sort(routing_matrix.getrow(lid).indices)
                flow_mask[desired_flows] = False
                actual_d_flow = int(np_sub_flow_list[desired_flows[0]] // num_paths_per_flow)
                which_sub_flow_active[actual_d_flow, flow_active_idx[actual_d_flow]] = 0
                flow_active_idx[actual_d_flow] = num_paths_per_flow
                for d_s_flow in desired_flows:
                    actual_d_s_flow = np_sub_flow_list[d_s_flow]
                    fid_to_subflow_removed[actual_d_flow] += 1
                    already_seen_subflows.add(actual_d_s_flow)
        else:
            min_link = np.where(rem_fair_share <= link_min_fair + constants.O_epsilon)
            assert np.all(min_link[0] < true_links.shape[0])
            # active_flows = np.unique(np.where(which_sub_flow_active)[0])
            curr_rate = link_min_fair * reshaped_active_sub_flow
            final_flow_rate[np_sub_flow_list] += curr_rate
            capacity_vector -= routing_matrix @ curr_rate
            pre_fid_to_subflow_remove = fid_to_subflow_removed.copy()
            for lid in min_link[0]:
                link_mask[lid] = False
                desired_flows = np.sort(routing_matrix.getrow(lid).indices)
                flow_mask[desired_flows] = False
                for d_s_flow in desired_flows:
                    actual_d_s_flow = np_sub_flow_list[d_s_flow]
                    already_seen_subflows.add(actual_d_s_flow)
                    actual_d_flow = int(actual_d_s_flow // num_paths_per_flow)
                    fid_to_subflow_removed[actual_d_flow] += 1
                    if which_sub_flow_active[actual_d_flow, actual_d_s_flow - actual_d_flow * num_paths_per_flow] == 0:
                        continue
                    # assert flow_active_idx[actual_d_flow] == actual_d_s_flow - actual_d_flow * num_paths_per_flow
                    which_sub_flow_active[actual_d_flow, flow_active_idx[actual_d_flow]] = 0

                    # flow_active_idx[actual_d_flow] += 1
                    # actual_d_s_flow += 1
                    while flow_active_idx[actual_d_flow] < num_paths_per_flow and \
                            actual_d_s_flow in already_seen_subflows:
                        flow_active_idx[actual_d_flow] += 1
                        actual_d_s_flow += 1

                    if flow_active_idx[actual_d_flow] >= num_paths_per_flow:
                        removed = np.count_nonzero(pre_fid_to_subflow_remove[:actual_d_flow] >= num_paths_per_flow)
                        # print(removed, starting_v_lid + actual_d_flow - removed, actual_d_flow, actual_d_s_flow, d_s_flow)
                        link_mask[starting_v_lid + actual_d_flow - removed] = False
                        continue

                    if path_exist_vector[num_paths_per_flow * actual_d_flow + flow_active_idx[actual_d_flow]] == 0:
                        removed = np.count_nonzero(pre_fid_to_subflow_remove[:actual_d_flow] >= num_paths_per_flow)
                        link_mask[starting_v_lid + actual_d_flow - removed] = False
                        while flow_active_idx[actual_d_flow] < num_paths_per_flow:
                            flow_active_idx[actual_d_flow] += 1
                            fid_to_subflow_removed[actual_d_flow] += 1
                            actual_d_s_flow += 1
                        continue
                    which_sub_flow_active[actual_d_flow, flow_active_idx[actual_d_flow]] = 1

        # print(np.count_nonzero(~link_mask[starting_v_lid:]), np.count_nonzero(~flow_mask) / 16)
        capacity_vector = np.compress(link_mask, capacity_vector)
        routing_matrix = routing_matrix[link_mask]
        routing_matrix = routing_matrix[:, flow_mask]
        np_sub_flow_list = np.compress(flow_mask, np_sub_flow_list)
        true_links = np.arange(true_links.shape[0] - np.count_nonzero(~link_mask[true_links]))
        # * diags(flow_mask)

    run_time_dict[COMPUTATION_TIME] = (datetime.now() - computation_st_time).total_seconds()

    dur = (datetime.now() - st_time).total_seconds()
    extract_st_time = datetime.now()
    flow_id_to_flow_rate_mapping = dict()
    for fid, (src, dst, demand) in list_commodities:
        sub_fid_list = np.arange(fid * num_paths_per_flow, (fid + 1) * num_paths_per_flow)
        flow_id_to_flow_rate_mapping[(src, dst)] = final_flow_rate[sub_fid_list]
    run_time_dict[EXTRACT_RATE] = (datetime.now() - extract_st_time).total_seconds()

    print(f"B4+ max-min fair;", dur)
    print(f"B4+ max-min fair;", run_time_dict)
    print(f"B4+ total allocated rate;", np.sum(final_flow_rate))
    print(f"B4+ num iterations;", iter_no)
    return flow_id_to_flow_rate_mapping, dur, run_time_dict
