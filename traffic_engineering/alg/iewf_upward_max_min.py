import copy
from collections import defaultdict
from datetime import datetime

import numpy as np
from scipy.sparse import csr_matrix

from utilities import constants
from ncflow.lib import problem


def get_routing_matrix(problem: problem.Problem, path_edge_idx_mapping, num_paths_per_flow, split_ratios, link_cap):
    fid_to_sub_fid_mapping = defaultdict(list)
    sub_fid_to_fid_mapping = dict()
    list_commodities = problem.sparse_commodity_list
    num_flows = len(list_commodities)
    num_links = len(problem.edges_list)
    capacity_vector = np.full(shape=(num_links + num_flows), fill_value=link_cap)
    sub_fid = 0
    max_link_id = num_links
    row_list = []
    column_list = []
    data_list = []
    for fid, (src, dst, demand) in list_commodities:
        pid_to_idx_mapping = path_edge_idx_mapping[src, dst]
        sub_fid_list = fid_to_sub_fid_mapping[fid]
        for pid, lid_list in pid_to_idx_mapping.items():
            row_list.append(lid_list)
            column_list.extend([sub_fid] * len(lid_list))
            data_list.extend([split_ratios[src, dst][pid]] * len(lid_list))
            sub_fid_list.append(sub_fid)
            sub_fid_to_fid_mapping[sub_fid] = fid
            sub_fid += 1
        column_list.extend([sfid for sfid in sub_fid_list])
        row_list.append(np.full(len(sub_fid_list), fill_value=max_link_id))
        data_list.extend([split_ratios[src, dst][sidx] for sidx in range(len(sub_fid_list))])
        capacity_vector[max_link_id] = demand
        max_link_id += 1
    row_list = np.concatenate(row_list)
    column_list = np.asarray(column_list, dtype=np.int64)
    data_list = np.asarray(data_list)
    routing_matrix = csr_matrix((data_list, (row_list, column_list)),
                                shape=(num_links + num_flows, sub_fid))
    return routing_matrix, capacity_vector, fid_to_sub_fid_mapping, sub_fid_to_fid_mapping


def get_exponential_decay_split_ratio(problem:problem.Problem, path_edge_mapping, base=0.1):
    split_ratios = dict()
    for fid, (src, dst, demand) in problem.sparse_commodity_list:
        num_paths = len(path_edge_mapping[src, dst])
        # total_split = np.sum([np.power(base, i) for i in range(num_paths)])
        coeff = np.array([np.power(base, i) for i in range(num_paths)])
        split_ratios[src, dst] = coeff / np.sum(coeff)
    return split_ratios


def get_rates(problem:problem.Problem, path_edge_mapping, link_cap, num_paths_per_flow, max_iter,
              initial_flow_rate=constants.EXPONENTIAL_DECAY, break_down=False):
    st_time = datetime.now()
    if initial_flow_rate == constants.EXPONENTIAL_DECAY:
        split_ratios = get_exponential_decay_split_ratio(problem, path_edge_mapping)
    else:
        raise Exception

    fid_to_total_flow_rate_mapping = defaultdict(lambda: [0] * num_paths_per_flow)

    for i in range(max_iter):
        routing_matrix, capacity_vector, fid_to_sub_fid_mapping, sub_fid_to_fid_mapping = get_routing_matrix(problem,
                                                                                                             path_edge_mapping,
                                                                                                             num_paths_per_flow,
                                                                                                             split_ratios,
                                                                                                             link_cap)

        fid_to_sub_fid_mapping_copy = copy.deepcopy(fid_to_sub_fid_mapping)
        demand_unit = 0.1
        num_flows = routing_matrix.shape[1]
        final_flow_rate = np.zeros(num_flows)
        per_flow_rate = np.zeros(num_flows)
        np_flow_list = np.arange(num_flows)

        while np_flow_list.shape[0]:
            print(f"iter {i} remaining flows {np_flow_list.shape[0]}")
            per_flow_rate += demand_unit
            current_util = routing_matrix @ per_flow_rate

            overloaded_link_mask = np.where(current_util >= capacity_vector - constants.O_epsilon)

            link_mask = np.full(routing_matrix.shape[0], fill_value=True)
            flow_mask = np.full(routing_matrix.shape[1], fill_value=True)
            for lid in overloaded_link_mask[0]:
                desired_flows = routing_matrix.getrow(lid).indices
                final_flow_rate[np_flow_list[desired_flows]] = routing_matrix[lid, desired_flows].multiply(per_flow_rate[desired_flows]).data
                normalize_factor = capacity_vector[lid] / np.sum(final_flow_rate[np_flow_list[desired_flows]])
                final_flow_rate[np_flow_list[desired_flows]] = final_flow_rate[np_flow_list[desired_flows]] * normalize_factor
                per_flow_rate[desired_flows] = per_flow_rate[desired_flows] * normalize_factor
                flow_mask[desired_flows] = False
                link_mask[lid] = False
                for sfid in desired_flows:
                    sfid_idx = np_flow_list[sfid]
                    fid = sub_fid_to_fid_mapping[sfid_idx]
                    sub_fid_list = fid_to_sub_fid_mapping[fid]
                    if sfid_idx in sub_fid_list:
                        sub_fid_list.remove(sfid_idx)
                        sub_fid_list_idx = np.searchsorted(np_flow_list, sub_fid_list)
                        assert np.all(np_flow_list[:-1] <= np_flow_list[1:])
                        per_flow_rate[sub_fid_list_idx] -= final_flow_rate[sfid_idx]
                        total_split = 1 - routing_matrix.getcol(sfid).data[0]
                        for o_fid_idx in sub_fid_list_idx:
                            routing_matrix[:, o_fid_idx] = routing_matrix[:, o_fid_idx] / total_split

            capacity_vector = np.compress(link_mask, capacity_vector)
            routing_matrix = routing_matrix[link_mask]
            capacity_vector -= routing_matrix @ (per_flow_rate * (1 - flow_mask))
            routing_matrix = routing_matrix[:, flow_mask]
            np_flow_list = np.compress(flow_mask, np_flow_list)
            per_flow_rate = np.compress(flow_mask, per_flow_rate)

        split_ratios = defaultdict(lambda: [0] * num_paths_per_flow)
        for fid, (src, dst, demand) in problem.sparse_commodity_list:
            total_flow = np.sum(final_flow_rate[fid_to_sub_fid_mapping_copy[fid]])
            for sidx, sub_fid in enumerate(fid_to_sub_fid_mapping_copy[fid]):
                fid_to_total_flow_rate_mapping[(src, dst)][sidx] = final_flow_rate[sub_fid]
                split_ratios[(src, dst)][sidx] = final_flow_rate[sub_fid] / total_flow
    dur = (datetime.now() - st_time).total_seconds()
    return fid_to_total_flow_rate_mapping, dur